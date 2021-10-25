source ./envs.bash
python3 -m bootleg_data_prep.merge_shuff_split \
    --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
    --subfolder_name full_wiki \
    --split 10
