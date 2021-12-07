echo
echo "=============================================================================="
echo "Step step4b-create-kg-adj"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/wikidata/create_kg_adj.py \
    --data $BOOTLEG_PREP_WIKIDATA_DIR \
    --out_dir wikidata_output \
    --processes $BOOTLEG_PREP_PROCESS_COUNT_MAX \
    --qids $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json
