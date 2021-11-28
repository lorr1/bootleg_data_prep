echo
echo "=============================================================================="
echo "Step step2a-map-titles-to-qids"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/wikidata/get_title_to_ids.py \
    --data $BOOTLEG_PREP_WIKIDATA_DIR \
    --wikipedia_xml $BOOTLEG_PREP_WIKIPEDIA_DUMP_FULL_FILENAME \
    --total_wikipedia_xml_lines 55721491 \
    --wikipedia_pageids $BOOTLEG_PREP_WIKIPEDIA_DIR/pageids/AA \
    --out_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/title_mappings
