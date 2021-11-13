echo
echo "=============================================================================="
echo "Step step3b-remove-bad-aliases"
echo "=============================================================================="
echo
source ./envs.bash
python3 -m bootleg_data_prep.remove_bad_aliases \
    --sentence_dir $BOOTLEG_WIKIPEDIA_DIR/sentences \
    --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
    --title_to_qid $BOOTLEG_WIKIPEDIA_DIR/title_mappings/title_to_all_ids.jsonl \
    --benchmark_qids '' \
    --processes $BOOTLEG_PROCESS_COUNT_IN_HIGH_MEMORY_LOAD
