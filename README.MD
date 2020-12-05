# This is a readme for generating training data from Wikipedia

Below is the process for data prep. At each stage, the run args are dumped into the data directory specified with timestamps. As this is training data and we only train on in KB QIDs, we drop aliases/mentions that aren't in our KB. Our process for benchmark data is different and in the `benchmarks` folder.

There are four required files for this process to run. (1) A Wikipedia dump that has been processed with `wiki_extractor.py` via the command

`python3 -u contextual_embeddings/data_prep/wiki_extractor.py --output sentences --output2 pageids enwiki-latest-pages-articles-multistream.xml`

This will output `sentences` and `pageids` folders for use in the rest of the process.

(2) You will need `title_to_all_ids.jsonl` (or some equivalent mapping of Wikipedia (old) page title, current Wikipedia page title, Wikidata title, QID, and Wikipedia page id).

(3) If you want to use Wikidata aliases in the process, you will need some alias mapping that goes from alias to list of QIDs. Otherwise, this map can be an empty json.

(4) The embeddings directory `embs` that stores QID to type ids and KG relationships. This is only used to generate slices based on types and KGs if desired.

### Step 1: Curate Aliases

This will read in all sentences from Wikipedia and gather all alias to QID mappings filtered by some prior score (aka an alias has to map to a QID at least X times). We gather anchor links and also map all page titles (including redirect) to the QIDs. We then merge these QIDs with those from Wikidata.

*Input*: sentence directory (`--sentence_dir`), Wikidata aliases (`--wd_aliases`), mapping of QIDs to Wikipedia titles (`--title_to_qid`)

*Output*: in `curate_aliases/` folder, there will be a variety of json outputs regarding what QIDs we filter and why. The necessary output is `alias_to_qid_filter.json` that is used in the next stage.

### Step 2: Remove Bad Aliases

This will read in all sentences from Wikipedia and use the previously build alias to QID mapping. This step will go through Wikipedia data and map anchor links to their QIDs (raw Wikipedia data uses titles). It will drop an anchor link if there is some span issue with the alias, the alias is empty (in the case of odd encoding issues leaving the alias empty), the alias isn't in our set, the title doens't have a QID, the QID is -1, or the QID isn't associated with that alias. We then build our first entity dump by including all QIDs seen in anchor links and all Wikipedia QIDs. We score each entity candidate for each alias based on the global entity popularity. 

*Input*: sentence directory (`--sentence_dir`), previous alias to QID map (`--alias_filter`), mapping of QIDs to Wikipedia titles (`--title_to_qid`)

*Output*: in `alias_filtered_sentences/` folder, there will be a variety of json outputs regarding what QIDs and aliases we filter and why. We also collect QID counts (`qid_counts.json`) and alias to QID counts (`alias_to_qid_count.json`) from the filtered data. There is also Wikipedia data chunks for the next stage.

### Step 3: Add Labels Single Func

This will read in the Wikipedia data from the last step along with entity dump and add mentions to sentences based on two heuristics. (1) If an alias for QIDX appears on QIDX's Wikipedia page, assume that alias points to QIDX. (2) If aliasY uniquely (or very often) points to QIDY on QIDX's Wikipedia page, add mentions of aliasY in the rest of the page and have them point to QIDY. Further, by default, we perform alias permutation where the aliases of added mentions are chosen to be highly conflicting aliases. To turn this off add (`--no_permute_alias`). If you do not want to use heuristic (2), add (--`no_coref`).

*Input*: data from step 2

*Output*: in `<subfolder_name>/` folder, there will be a variety of json outputs regarding QID (`filtered_qid_count.json`) counts, alias to QID (`filtered_aliases_to_qid_count.json`) counts from the weakly labelled data, QID (`filtered_qids_to_qids.json`) and alias (`filtered_aliases_to_alias.json`) cooccurrence counts, as well. There is also augmented Wikipedia data chunks.

### Step 4: Pronoun Coref

This will read in the Wikipedia data form the last step and weakly label pronouns in the data if the pronoun matches the gender of the QID whose page we are on. This is only applicable to people Wikipedia pages. If the `--swap_titles` flag is turned on, the pronoun text in the sentence will be swapped with the QID page title. We use the alias associated with the QID that is the most conflicting (has the most candidates). This is run via `python3 -m bootleg_data_prep.prn_labels <input_dir> <output_dir> <input_dir>/entity_db/entity_mappings` where the last path is to the entity dump you want to use. 

*Input*: data from step 3

*Output*: the same Wikipedia chunks with pronouns will be in `<output_dir>`.

### Step 5: Filter Data

This will simply filter the data based on some function you define in `my_filter_funcs.py` and truncate QIDs based on the maximum number of candidates provided. After this, it then goes through and filters the entities and alias to QID mappings to only be ones that are seen in this data or from a list of benchmark QIDs provided (This ensures we keep a row for all benchmark QIDs. If the QID is never seen during training, the learned embedding will stay random.). It also truncates based on maximum candidates. Finally, it filters the data one more time to only include QIDs that are candidates for an alias (truncating at maxinum candidates may drop some), if the train in candidates param is set. ALso, it computes global statistics over the data that will be used in slice generation.

*Input*: data from step 4

*Output*: in `<subfolder_name>/` folder, there will filtered output data. In `<subfolder_name>_stats/` folder, there will be two sets of statistics. One calcuated over all mentions (with aug), and one over just mentions from the original Wikipedia data.

### Step 6: Merge and Shuffle

This reads in all data, shuffles it, and outputs it to train, test, and dev folders.

*Input*: data from step 5

*Output*: in `<subfolder_name>/` folder, there will now be train, test, and dev split folders.

### Step 7: Slice Generation Prep (part 1 and part 2)
This reads in all the training data and computes bag of words for each gold entity seen. We use this as a filter in the next step.

*Input*: data from step 6

*Output*: in `<subfolder_name>/entity_dump` folder, there will be a bag of words folder that stores the Marisa Trie. 

### Step 8: Slice Generation

This reads in all data and generates slices based on the slice definitions. It allows for only subsets of all files from step 5 to be read in for testing and for distributed processing.

*Input*: data from step 7

*Output*: in `<subfolder_name>_final/` folder, there will now be train, test, and dev files along with slices of dev and test.

### Example Script
```
DATASET = "USA" # the name of the dataset you want to generate
FILTER_FUNC = "sentence_filterUsaAlias" # the corresponding filter function to create this dataset 
MAX_CANDS = 30 
PYTHON = python3.6
$PYTHON -m bootleg_data_prep.curate_aliases --min_frequency 2
$PYTHON -m bootleg_data_prep.remove_bad_aliases --processes 15 --benchmark_qids contextual_embeddings/data_prep/benchmarks/all_benchmark_ids.json
$PYTHON -m bootleg_data_prep.add_labels_single_func --processes 25
$PYTHON -m bootleg_data_prep.prn_labels data/wiki_dump/orig_wl data/wiki_dump/orig data/wiki_dump/orig_nocoref/entity_db/entity_mappings --num_workers 50
$PYTHON -m bootleg_data_prep.data_filter --processes 30 --train_in_candidates --sentence_filter_func $FILTER_FUNC --subfolder_name $DATASET --max_candidates $MAX_CANDS --benchmark_qids contextual_embeddings/data_prep/benchmarks/all_benchmark_ids.json
$PYTHON -m bootleg_data_prep.merge_shuff_split --subfolder_name $DATASET --split 10
$PYTHON -m bootleg_data_prep.prep_generate_slices_part1 --subfolder_name $DATASET --processes 90
$PYTHON -m bootleg_data_prep.prep_generate_slices_part2 --subfolder_name $DATASET --processes 90 --overwrite
$PYTHON -m bootleg_data_prep.generate_slices --processes 40  --subfolder_name $DATASET --overwrite
```

The generate slices steps requires the `embs/` directory. See the arg parser details for help.