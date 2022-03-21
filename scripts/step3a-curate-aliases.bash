echo
echo "=============================================================================="
echo "Step step3a-curate-aliases"
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
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/curate_aliases.py \
    --min_frequency 2 \
    --sentence_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/sentences \
    --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
    --wd_aliases $BOOTLEG_PREP_WIKIPEDIA_DIR/augmented_alias_map_large.jsonl \
    --title_to_qid $BOOTLEG_PREP_WIKIPEDIA_DIR/title_mappings/title_to_all_ids.jsonl \
    $STRIP_P $LOWER_P \
    --processes $BOOTLEG_PREP_PROCESS_COUNT_MAX
