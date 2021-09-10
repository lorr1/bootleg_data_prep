source ./envs.bash
python3 -m bootleg_data_prep.data_filter \
    --processes $BOOTLEG_PROCESS_COUNT \
    --train_in_candidates \
    --subfolder_name full_wiki \
    --max_candidates 30 \
    --sentence_filter_func false_filter \
    --orig_dir alias_filtered_sentences \
    --no_filter_entities_data \
    --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
    --filter_file '' --benchmark_qids ''
