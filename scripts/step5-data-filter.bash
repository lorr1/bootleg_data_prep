echo
echo "=============================================================================="
echo "Step step6-merge-stuff-split"
echo "=============================================================================="
echo
source ./envs.bash
if [ "$BOOTLEG_PREP_WEAK_LABELING" = true ] ; then
  ORIG_DIR=weak_labeling_data
else
  ORIG_DIR=alias_filtered_sentences
fi
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/data_filter.py \
    --processes $BOOTLEG_PROCESS_COUNT_MAX \
    --processes_in_memory_load $BOOTLEG_PROCESS_COUNT_MIN \
    --train_in_candidates \
    --subfolder_name full_wiki \
    --max_candidates 30 \
    --sentence_filter_func false_filter \
    --orig_dir $ORIG_DIR \
    --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
    --filter_file '' --benchmark_qids ''
