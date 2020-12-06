## Installation
To run this code, you'll need to install the `requirements.txt` and run

`python3 setup.py develop`.

You will also need access to https://github.com/neelguha/wikidata-processor. Contact Laurel for getting access to this.

## Running
For all these instructions, my working directory was `/lfs/raiders8/0/lorr1/`. Replace this with your working directory. Note that the term "mention" and "alias" are used interchangably. We have more detailed descriptions of what each of the following functions do in the main folder's `README.MD`.

0. _Download and extract wikidata_ (place in `/lfs/raiders8/0/lorr1/wikidata/raw_data/latest-all.json`) and process. This will allow us to more easily query for metadata.

Expected Time: 15 hours
`python3 -m processor.process_dump --input_file /lfs/raiders8/0/lorr1/wikidata/raw_data/latest-all.json --out_dir /lfs/raiders8/0/lorr1/wikidata_sv/processed_batches --language_id sv --num_processes 40`

1. _Make working folder and download wikipedia._
```
mkdir sv_data
cd sv_data
wget <WIKIPEDIA DUMP>
ls
>>> svwiki-20201120-pages-articles-multistream.xml
```
Then run the wiki extractor code to parse Wikipedia.

Expected Time: 4 hours
`python3 -m bootleg_data_prep.wiki_extractor --output sentences --output2 pageids svwiki-20201120-pages-articles-multistream.xml --processes 5 &> wiki_extractor.out`

This will output
```
INFO: Finished 5-process extraction of 3552747 articles in 14728.0s (241.2 art/s)
INFO: total of page: 3988933, total of articl page: 3552747; total of used articl page: 3552747
Final end time 14728.023938179016s
```

We expect `ls sv_data` to output
`pageids  sentences  svwiki-20201120-pages-articles-multistream.xml`

2. _Get mapping of all wikipedia ids to QIDs._

Expected Time: 20 minutes
`python3 -m processor.bootleg.get_title_to_ids \
    --data /lfs/raiders8/0/lorr1/wikidata_sv \
    --wikipedia_xml /lfs/raiders8/0/lorr1/sv_data/svwiki-20201120-pages-articles-multistream.xml \
    --total_wikipedia_xml_lines 436717867 \
    --wikipedia_pageids /lfs/raiders8/0/lorr1/sv_data/pageids/AA \
    --out_dir /lfs/raiders8/0/lorr1/sv_data/title_mappings`

You should now have `title_mappings` in `sv_data`.

(Optional) If you want to add wikidata aliases and associated QIDs to our candidate lists, run the following.

Expected Time: less than 1 hour
`python3 -m processor.bootleg.create_aliases \
    --qids '' \
    --data /lfs/raiders8/0/lorr1/wikidata_sv \
    --out_file /lfs/raiders8/0/lorr1/sv_data/augmented_alias_map_large.jsonl`
    
If you run this, you should now have `augmented_alias_map_large.jsonl` in `sv_data`.

3. _Curate alias and candidate mappings._ We mine aliases from Wikipedia hyperlinks. The `min_frequency` param controls the number of times and alias needs to be seen with an entity to count as a potential candidate. We also map all page titles (including redirect) to the QIDs and then merge these QIDs with those from Wikidata.

*Input*: sentence directory (`--sentence_dir`), Wikidata aliases (`--wd_aliases`), mapping of QIDs to Wikipedia titles (`--title_to_qid`)

*Output*: in `/lfs/raiders8/0/lorr1/sv_data/data/wiki_dump/curate_aliases/` folder, there will be a variety of json outputs regarding what QIDs we filter and why. The necessary output is `alias_to_qid_filter.json` that is used in the next stage.

The next step is to remove bad mentions. We will read in all sentences from Wikipedia and use the previously build alias to QID mapping. This step will go through Wikipedia data and map anchor links to their QIDs. It will drop an anchor link if there is some span issue with the alias, the alias is empty (in the case of odd encoding issues leaving the alias empty), the alias isn't in our set, the title doens't have a QID, the QID is -1, or the QID isn't associated with that alias. We then build our first entity dump by including all QIDs seen in anchor links and all Wikipedia QIDs. We score each entity candidate for each alias based on the global entity popularity.  

*Input*: sentence directory (`--sentence_dir`), previous alias to QID map (`--alias_filter`), mapping of QIDs to Wikipedia titles (`--title_to_qid`)

*Output*: in `/lfs/raiders8/0/lorr1/sv_data/data/wiki_dump/alias_filtered_sentences/` folder, there will be a variety of json outputs regarding what QIDs and aliases we filter and why. We also collect QID counts (`qid_counts.json`) and alias to QID counts (`alias_to_qid_count.json`) from the filtered data. There is also Wikipedia data chunks for the next stage.

Expected Time: less than 1 hour
- `python3 -m bootleg_data_prep.curate_aliases \
    --min_frequency 2 \
    --sentence_dir /lfs/raiders8/0/lorr1/sv_data/sentences \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --wd_aliases /lfs/raiders8/0/lorr1/sv_data/augmented_alias_map_large.jsonl \
    --title_to_qid /lfs/raiders8/0/lorr1/sv_data/title_mappings/title_to_all_ids.jsonl \
    --processes 10`

- `python3 -m bootleg_data_prep.remove_bad_aliases \
    --sentence_dir /lfs/raiders8/0/lorr1/sv_data/sentences \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --title_to_qid /lfs/raiders8/0/lorr1/sv_data/title_mappings/title_to_all_ids.jsonl \
    --benchmark_qids '' \
    --processes 10`

4. _Extract KG and Types._ This extract Wikidata metadata for Bootleg models.

Expected Time: less than 1 hour.
`python3 -m processor.bootleg.get_all_wikipedia_triples \
    --data /lfs/raiders8/0/lorr1/wikidata_sv \
    --out_dir wikidata_output \
    --processes 10 \
    --qids /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json`
`python3 -m processor.bootleg.create_kg_adj \
    --data /lfs/raiders8/0/lorr1/wikidata_sv \
    --out_dir wikidata_output \
    --processes 10 \
    --qids /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json`
`python3 -m processor.bootleg.get_types \
    --data /lfs/raiders8/0/lorr1/wikidata_sv \
    --out_dir wikidata_output \
    --processes 10 \
    --qids /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json`

Then run
```
mkdir /lfs/raiders8/0/lorr1/sv_data/embs
cp /lfs/raiders8/0/lorr1/wikidata_sv/wikidata_output/kg_triples.txt /lfs/raiders8/0/lorr1/sv_data/embs/kg_triples_1129.txt
cp /lfs/raiders8/0/lorr1/wikidata_sv/wikidata_output/kg_adj.txt /lfs/raiders8/0/lorr1/sv_data/embs/kg_adj_1129.txt
cp /lfs/raiders8/0/lorr1/wikidata_sv/wikidata_output/wikidatatitle_to_typeqid.json /lfs/raiders8/0/lorr1/sv_data/embs/wikidatatitle_to_typeqid_1129.json
cp /lfs/raiders8/0/lorr1/wikidata_sv/wikidata_output/wikidatatitle_to_typeid.json /lfs/raiders8/0/lorr1/sv_data/embs/wikidatatitle_to_typeid_1129.json
cp /lfs/raiders8/0/lorr1/wikidata_sv/wikidata_output/wikidata_types.json /lfs/raiders8/0/lorr1/sv_data/embs/wikidata_types_1129.json
```
Once things are in place, you can generate our final relation type mapping. The `pid_names.json` file can be found in `utils/param_files`.
- `python3 -m processor.bootleg.get_relation_types_from_triples \
    --pid_file utils/param_files/pid_names.json \
    --kg_triples /lfs/raiders8/0/lorr1/sv_data/embs/kg_triples_1129.txt \
    --type_file /lfs/raiders8/0/lorr1/sv_data/embs/wikidata_types_1129.json \
    --output_file /lfs/raiders8/0/lorr1/sv_data/embs/kg_relation_types_1129.json \
    --output_vocab_file /lfs/raiders8/0/lorr1/sv_data/embs/relation_to_typeid_1129.json`

At the end, you should have
`ls sv_data/embs`
`kg_adj_1129.txt  kg_relation_types_1129.json  kg_triples_1129.txt  relation_to_typeid_1129.json  wikidatatitle_to_typeid_1129.json  wikidatatitle_to_typeqid_1129.json  wikidata_types_1129.json`

Note that if you'd like to filter the type system or something else, you can. We also support adding a coarser grained type system (we use Hyena) if desired. They are left blank by default.

(Optional) _Weak labeling._ Our weak labelling pipeline is still being iterated on and optimized. We support labelling pronouns and alternate names of entities. These are the files `add_labels_single_func.py` and `prn_labels.py`. If you'd like to use these, let Laurel know. We hope to release optimized and more modifiable versions of these files by the end of Dec.

# Add alternate names will read in the Wikipedia data from the last step along with entity dump and add mentions to sentences based on two heuristics. (1) If an alias for QIDX appears on QIDX's Wikipedia page, assume that alias points to QIDX. (2) If aliasY uniquely (or very often) points to QIDY on QIDX's Wikipedia page, add mentions of aliasY in the rest of the page and have them point to QIDY. Further, by default, we perform alias permutation where the aliases of added mentions are chosen to be highly conflicting aliases. To turn this off add (`--no_permute_alias`). If you do not want to use heuristic (2), add (--`no_coref`).

# *Input*: data from step 2

# *Output*: in `<subfolder_name>/` folder, there will be a variety of json outputs regarding QID (`filtered_qid_count.json`) counts, alias to QID (`filtered_aliases_to_qid_count.json`) counts from the weakly labelled data, QID (`filtered_qids_to_qids.json`) and alias (`filtered_aliases_to_alias.json`) cooccurrence counts, as well. There is also augmented Wikipedia data chunks.

# Pronoun coref will read in the Wikipedia data form the last step and weakly label pronouns in the data if the pronoun matches the gender of the QID whose page we are on. This is only applicable to people Wikipedia pages. If the `--swap_titles` flag is turned on, the pronoun text in the sentence will be swapped with the QID page title. We use the alias associated with the QID that is the most conflicting (has the most candidates). This is run via `python3 -m bootleg_data_prep.prn_labels <input_dir> <output_dir> <input_dir>/entity_db/entity_mappings` where the last path is to the entity dump you want to use. 

# *Input*: data from step 3

# *Output*: the same Wikipedia chunks with pronouns will be in `<output_dir>`.



5. _Filter data._
This will flatten the document data we've used this far into a sentences and remove sentences that pass some filtering criteria. The `false_filter` means we remove no sentences from the training data but arbitrary filters can be added in `my_filter_funcs.py`. This also will filter the entity dump so that QIDs are removed if they are not a candidate for any alias of any gold QID in the dataset. These QIDs are essentially unseen during training so we remove them to reduce memory requirements of the model. We can add these back afterwards by assigning them the embedding of some rare entity. The `disambig_qids` and `benchmark_qids` params are for removing disambiguation pages and making sure all benchmark QIDs are kept in the dump. We lastly truncate candidate lists to only have `mac_candidates` candidates.

*Input*: data from step 4

*Output*: in `/lfs/raiders8/0/lorr1/sv_data/data/<subfolder_name>/` folder, there will filtered output data. In `/lfs/raiders8/0/lorr1/sv_data/data/<subfolder_name>_stats/` folder, there will be two sets of statistics. One calculated over all mentions (with weak labeling), and one over just mentions from the original Wikipedia data (gold).

Expected Time: less than 1 hour.
# subfolder_name is whatever you want to call the folder of the data
`python3 -m bootleg_data_prep.data_filter \
    --processes 10 \
    --train_in_candidates \
    --subfolder_name full_wiki \
    --max_candidates 30 \
    --sentence_filter_func false_filter \
    --orig_dir alias_filtered_sentences \
    --no_filter_disambig_entities \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --disambig_file '' --benchmark_qids ''`

6. Split data.
This will split data by Wikipedia pages (by default). With `--split 10`, 10% of pages will go to test, 10% to dev, and 80% to train.

*Input*: data from step 5

*Output*: in `/lfs/raiders8/0/lorr1/sv_data/data/<subfolder_name>/` folder, there will now be train, test, and dev split folders.

Expected Time: 30 minutes.
`python3 -m bootleg_data_prep.merge_shuff_split \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --subfolder_name full_wiki \
    --split 10`

7. Slice data (optional). This prepares our data for fast generation of slices that can be used to monitor the model while it's running. The prep steps read in all the training data and computes bag of words for each gold type and relation seen and filters by TFIDF (used for affordance slice). The final step is the fastest for quick iteration on slice definitions.

*Input*: data from step 6

*Output*: in `/lfs/raiders8/0/lorr1/sv_data/data/wiki_dump/<subfolder_name>/entity_dump` folder, there will be a bag of words folder that stores the Marisa Trie for the TFIDF collections. 

Out final step reads in all data and generates slices based on the slice definitions in `utils/slice_definitions.py` and the slice function in `generate_slices.py`.

*Input*: data from step 6

*Output*: in `/lfs/raiders8/0/lorr1/sv_data/data/wiki_dump/<subfolder_name>_0_-1_final/` folder, there will now be train, test, and dev files along with slices of dev and test.

Expected Time: 4 hours.
- `python3 -m bootleg_data_prep.prep_generate_slices_part1 \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --subfolder_name full_wiki \
    --processes 10 \
    --multilingual`
- `python3 -m bootleg_data_prep.prep_generate_slices_part2 \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --subfolder_name full_wiki \
    --processes 10 \
    --emb_dir /lfs/raiders8/0/lorr1/sv_data/embs \
    --kg_adj kg_adj_1129.txt \
    --kg_triples kg_triples_1129.txt \
    --hy_vocab ''\
    --hy_types ''\
    --wd_vocab wikidatatitle_to_typeid_1129.json \
    --wd_types wikidata_types_1129.json \
    --rel_vocab relation_to_typeid_1129.json \
    --rel_types kg_relation_types_1129.json \
    --multilingual`
- `python3 -m bootleg_data_prep.generate_slices \
    --subfolder_name full_wiki \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --processes 10 \
    --emb_dir /lfs/raiders8/0/lorr1/sv_data/embs \
    --kg_adj kg_adj_1129.txt \
    --kg_triples kg_triples_1129.txt \
    --hy_vocab ''\
    --hy_types ''\
    --wd_vocab wikidatatitle_to_typeid_1129.json \
    --wd_types wikidata_types_1129.json \
    --rel_vocab relation_to_typeid_1129.json \
    --rel_types kg_relation_types_1129.json`
    
    
The last step is to move the data to where you're going to run models.

`cp -r sv_data/data/wiki_dump/full_wiki_final_-1_0 <path_to_model_training_data>/full_wiki_train`

## Run Bootleg Model
See the Bootleg repo for more instructions. An example config is below to run Bootleg using our `emmental` branch. For speed, you might want to sample data to make the test and dev datasets smaller for eval during training (see `bootleg_emmental/utils/preprocessing/sample_eval_data.py`). See the README on the `emmental` branch for setup instructions. Comments in the YAML file indicate what you need to fill out.

Note if you are using a different language, you might want to use a different BERT backend.

```yaml
emmental:
  lr: 1e-4
  n_epochs: 10
  evaluation_freq: 1.0
  log_path: logs
  checkpointing: true
  checkpoint_all: true
  checkpoint_freq: 1
run_config:
  eval_batch_size: 256
  dataloader_threads: 4
  dataset_threads: 50
  spawn_method: forkserver
train_config:
  batch_size: 96
model_config:
  hidden_size: 256
  num_heads: 16
  num_model_stages: 2
  ff_inner_size: 1024
  attn_class: BootlegM2E
data_config:
  data_dir: <path to full_wiki_train>  # FILL OUT
  emb_dir: <path to sv_data/embs> # FILL OUT
  ent_embeddings:
       - key: learned
         load_class: LearnedEntityEmb
         freeze: false
         args:
           learned_embedding_size: 256
           mask_perc: 0.0  # CONSTANT REGULARIZATION
#           regularize_mapping: /dfs/scratch0/lorr1/bootleg/data/wiki_0906_pg_emm/qid2reg_2.csv # INVERSE REGULARIZATION - SEE bootleg_emmental/utils/preprocessing/build_regularization_mapping.py
       - key: learned_type_wiki
         load_class: LearnedTypeEmb
         freeze: false
         args:
           type_labels: wikidata_types_0905.json # TYPES INSIDE EMBS FOLDER
           max_types: 3
           type_dim: 128
           merge_func: addattn
           attn_hidden_size: 128
       - key: learned_type_relations
         load_class: LearnedTypeEmb
         freeze: false
         args:
           type_labels: kg_relation_types_0905.json # TYPES INSIDE EMBS FOLDER
           max_types: 3
           type_dim: 128
           merge_func: addattn
           attn_hidden_size: 128
       - key: adj_index
         load_class: KGIndices
         batch_on_the_fly: true
         dtype: int16
         args:
           kg_adj: kg_adj_0905.txt
  entity_dir: <path to entity_dir INSIDE of full_wiki_train>  # FILL OUT
  max_aliases: 10
  max_seq_len: 100
  overwrite_preprocessed_data: false
  dev_dataset:
    file: dev.jsonl
    use_weak_label: true
  test_dataset:
    file: test.jsonl
    use_weak_label: true
  train_dataset:
    file: train.jsonl
    use_weak_label: true
  train_in_candidates: true
  word_embedding:
    cache_dir: pretrained_bert_models # THIS WILL BE GENERATED IF IT DOESN'T EXIST OTHERWISE
    freeze: true
    layers: 12
```




