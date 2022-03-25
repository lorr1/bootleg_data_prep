## Installation
To run this code, you will need to install Bootleg

```
git clone git@github.com:HazyResearch/bootleg bootleg
cd bootleg
python3 setup.py install
cd ..
```

and then run

```
git clone git@github.com:lorr1/bootleg_data_prep.git bootleg_data_prep
cd bootleg_data_prep
pip install -r requirements.txt
python3 setup.py develop
cd ..
```


## Setup
To setup all the steps, you must modify `scripts/local_envs/set_my_env_vars.bash` to set the environment variables that you desire. Then, before you run any of the steps below, you will need to run `scripts/local_envs/setx.bash` (or otherwise set these environment variables). Lastly, before the very first time running these scripts, run `scripts/setup.bash`. This will first try to install the following packages for linux 
```
sudo apt install aria2 lbzip2 pv pigz
```
It will then create all relevant folders. You can skip this script for future runs.

## Running
Each step will call `scripts/envs.bash` to setup all related variables. If you don't properly setup the environment variables from `Setup`, this will break.

The entire process is a sequence of 8 steps that start from downloading the data to processing the files for bootleg models (all steps reside in `scripts`). Some steps have multiple subparts. The `scripts/all.bash` scripts has all steps all together. We go over what each step is doing below for reference. Run each script with `bash <script_name>`.

#### Step 0
This downloads wikidata and extracts it using `simple-wikidata-db` (modified from https://github.com/neelguha/simple-wikidata-db). This requires the correct langauge code for parsing.

#### Step 1
This download wikipedia and processes the data from the WikiExtractor from [here](https://attardi.github.io/wikiextractor/). The last step parses the extractor output into two folders: `sentences` and `pageids`.

#### Step 2
(a) Get mapping of all wikipedia ids to QIDs. I manually set to `total_wikipedia_xml_lines` for progress bars via the `wc -l` command, but this is not required.

(b) Adds wikidata aliases and associated QIDs to our candidate lists.

#### Step 3
(a) Curates alias and candidate mappings. We mine aliases from Wikipedia hyperlinks. The `min_frequency` param controls the number of times and alias needs to be seen with an entity to count as a potential candidate. We also map all page titles (including redirect) to the QIDs and then merge these QIDs with those from Wikidata.

(b) The next step is to remove bad mentions. We will read in all sentences from Wikipedia and use the previously build alias to QID mapping. This step will go through Wikipedia data and map anchor links to their QIDs. It will drop an anchor link if there is some span issue with the alias, the alias is empty (in the case of odd encoding issues leaving the alias empty), the alias isn't in our set, the title doens't have a QID, the QID is -1, or the QID isn't associated with that alias. We then build our first entity dump by including all QIDs seen in anchor links and all Wikipedia QIDs. We score each entity candidate for each alias based on the global entity popularity.  
    
#### Step 4
This extract Wikidata KGs, types, and descriptions for Bootleg models.

Note step 4d and 4e are experimental. By default, they are turned off. These two steps can be skipped. This first step is our weak labelling pipeline. We support labelling pronouns and alternate names of entities. These are the files `add_labels_single_func.py` and `prn_labels.py`. The second step is another form of weak labeling where we label other mentions on the page based on aliases for the QID of that page we are on.

#### Step 5
This will flatten the document data we've used this far into a sentences and remove sentences that pass some filtering criteria. The `false_filter` means we remove no sentences from the training data but arbitrary filters can be added in `my_filter_funcs.py`. This also will filter the entity dump so that QIDs are removed if they are not a candidate for any alias of any gold QID in the dataset. These QIDs are essentially unseen during training so we remove them to reduce memory requirements of the model. We can add these back afterwards by assigning them the embedding of some rare entity. The `disambig_qids` and `benchmark_qids` params are for removing disambiguation pages and making sure all benchmark QIDs are kept in the dump. We lastly truncate candidate lists to only have `max_candidates` candidates.

#### Step 6
This will split data by Wikipedia pages (by default). With `--split 10`, 10% of pages will go to test, 10% to dev, and 80% to train.

#### Step 7
We create the final entity dump for all QID data and metadata. This requires a mapping from PID to string names. We have downloaded some with scrapers in `utils/param_files/pid_names_LANGCODE.json`. If your language isn't there, you can also try to scrape them from https://w.wiki/4xtZ using the query

```
SELECT ?property ?propertyLabel ?propertyDescription (GROUP_CONCAT(DISTINCT(?altLabel); separator = ", ") AS ?altLabel_list) WHERE {
    ?property a wikibase:Property .
    OPTIONAL { ?property skos:altLabel ?altLabel . FILTER (lang(?altLabel) = "zh") }
    SERVICE wikibase:label { bd:serviceParam wikibase:language "zh" .}
 }
GROUP BY ?property ?propertyLabel ?propertyDescription
LIMIT 10000
```
where you replace the language code to be the appropriate one.

#### Step 8
This copies the final data folder to a desired location.

## Run Bootleg Model
See the Bootleg docs at https://bootleg.readthedocs.io/en/latest/index.html.

## Before Deployment
One last step before deployment is to filter aliases that are cleaner for deployment. The code for these steps are in the Bootleg repo, but we list instructions here.

Run
```
python3 -m bootleg.utils.preprocessing.comput_statistics --lower --data_dir <PATH TO TRAINING DATA> --num_workers 10
```
(if you set strip to be True, make sure to add the `--strip` command. Remove them if you set those environment variables to False). This will compute a bunch of stats over the training data. And, by default, save in the data_dir.

Then run `scripts/final_filter_alias_cand_map.ipynb`. This will load the alias_text_counts.json file and alias_counts.json from the previous step (which computes the number of times an alias appears in text independent of it’s an entity or not) and look at all aliases from the profile. If an alias appeared frequently in text as not an alias, we would remove it.

The final saved `entity_db` folder from that notebook should be used as the `entity_dir` in the bootleg config when deploying the model.