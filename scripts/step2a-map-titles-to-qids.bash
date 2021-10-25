source ./envs.bash
python3 -m bootleg_data_prep.wikidata.get_title_to_ids \
    --data $BOOTLEG_WIKIDATA_DIR \
    --wikipedia_xml $BOOTLEG_WIKIPEDIA_DIR/$BOOTLEG_WIKIPEDIA_DUMP_FILENAME \
    --total_wikipedia_xml_lines 55721491 \
    --wikipedia_pageids $BOOTLEG_WIKIPEDIA_DIR/pageids/AA \
    --out_dir $BOOTLEG_WIKIPEDIA_DIR/title_mappings
