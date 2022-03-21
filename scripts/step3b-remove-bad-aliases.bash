echo
echo "=============================================================================="
echo "Step step3b-remove-bad-aliases"
echo "=============================================================================="
echo
source ./envs.bash
if [ "$STRIP_PUNC_ALIASES" = true ] ; then
  STRIP_P=""
else
  STRIP_P="--not_strip"
fi
if [ "$LOWER_CASE_ALIASES" = true ] ; then
  LOWER_P=""
else
  LOWER_P="--not_strip"
fi
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/remove_bad_aliases.py \
    --sentence_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/sentences \
    --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
    --title_to_qid $BOOTLEG_PREP_WIKIPEDIA_DIR/title_mappings/title_to_all_ids.jsonl \
    --benchmark_qids '' \
    $STRIP_P $LOWER_P \
    --processes $BOOTLEG_PREP_PROCESS_COUNT_MAX
