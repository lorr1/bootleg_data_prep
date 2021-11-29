echo
echo "=============================================================================="
echo "Step step3b-remove-bad-aliases"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/remove_bad_aliases.py \
    --sentence_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/sentences \
    --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
    --title_to_qid $BOOTLEG_PREP_WIKIPEDIA_DIR/title_mappings/title_to_all_ids.jsonl \
    --benchmark_qids '' \
    --processes $BOOTLEG_PROCESS_COUNT_MIN
