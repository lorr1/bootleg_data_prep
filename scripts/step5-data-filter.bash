echo
echo "=============================================================================="
echo "Step step6-merge-stuff-split"
echo "=============================================================================="
echo
source ./envs.bash
if test ! -z "$USE_WEAK_LABELING"; then
  ORIG_DIR=weak_labeling_data
else
  ORIG_DIR=alias_filtered_sentences
fi
python3 -m bootleg_data_prep.data_filter \
    --processes $BOOTLEG_PROCESS_COUNT \
    --processes_in_memory_load $BOOTLEG_PROCESS_COUNT_IN_HIGH_MEMORY_LOAD \
    --train_in_candidates \
    --subfolder_name full_wiki \
    --max_candidates 30 \
    --sentence_filter_func false_filter \
    --orig_dir $ORIG_DIR \
    --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
    --filter_file '' --benchmark_qids ''
