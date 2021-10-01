source ./envs.bash
python3 -m bootleg_data_prep.wikidata.create_aliases \
    --data $BOOTLEG_WIKIDATA_DIR \
    --out_file $BOOTLEG_WIKIPEDIA_DIR/augmented_alias_map_large.jsonl

