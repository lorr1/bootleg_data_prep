source ./envs.bash
python3 -m bootleg_data_prep.wikidata.get_entity_descriptions \
    --data $BOOTLEG_WIKIDATA_DIR \
    --out_dir wikidata_output \
    --processes $BOOTLEG_PROCESS_COUNT \
    --wikipedia_page_data $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump/alias_filtered_sentences \
    --qids $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json
