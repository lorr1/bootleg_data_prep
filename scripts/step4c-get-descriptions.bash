echo
echo "=============================================================================="
echo "Step step4c-get-descriptions"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/wikidata/get_entity_descriptions.py \
    --data $BOOTLEG_PREP_WIKIDATA_DIR \
    --out_dir wikidata_output \
    --wikipedia_page_data $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/alias_filtered_sentences \
    --qids $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json
