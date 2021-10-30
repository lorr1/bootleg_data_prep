source ./envs.bash
python3 -m bootleg_data_prep.curate_aliases \
    --min_frequency 2 \
    --sentence_dir $BOOTLEG_WIKIPEDIA_DIR/sentences \
    --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
    --wd_aliases $BOOTLEG_WIKIPEDIA_DIR/augmented_alias_map_large.jsonl \
    --title_to_qid $BOOTLEG_WIKIPEDIA_DIR/title_mappings/title_to_all_ids.jsonl \
    --processes $BOOTLEG_PROCESS_COUNT
