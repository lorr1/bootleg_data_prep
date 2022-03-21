echo
echo "=============================================================================="
echo "Step step4b-get-types"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/wikidata/get_types.py \
    --data $BOOTLEG_PREP_WIKIDATA_DIR \
    --out_dir wikidata_output \
    --processes $BOOTLEG_PREP_PROCESS_COUNT_MAX \
    --qids $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json
