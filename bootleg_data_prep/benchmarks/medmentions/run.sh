#!/bin/bash

set -e

scriptDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# assume to run from the root of bootleg
pip install scispacy=0.2.4
wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/umls_2017_aa_cat0129.json
SCISPACY_UMLS_FILE="umls_2017_aa_cat0129.json"
UMLS_REL_FILE='/home/xling/data/health/2017AA-active/2017aa_cat0129_rels.jsonl'

MEDMENTIONS_DATA_PATH=${MEDMENTIONS_DATA_PATH:-"/home/xling/data/health/MedMentions/"}
OUTPUT_PATH=${OUTPUT_PATH:-"/home/xling/data/health/MedMentions/prep/"}
DATASET=${DATASET:-"full"}  # st21pv

python -m bootleg_data_prep.medmentions prep-text-data $MEDMENTIONS_DATA_PATH $OUTPUT_PATH $DATASET
python -m bootleg.utils.gen_entity_mappings --entity_dir ${OUTPUT_PATH}/${DATASET}/entity_data/ --qid2title ${OUTPUT_PATH}/${DATASET}/entity_data/qid2title.json --alias2qids ${OUTPUT_PATH}/${DATASET}/entity_data/alias2qids_scispacy.json

python -m bootleg_data_prep.medmentions prep-entity-data $OUTPUT_PATH $DATASET $SCISPACY_UMLS_FILE $UMLS_REL_FILE
