echo
echo "=============================================================================="
echo "Step step2b-create-aliases"
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
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/wikidata/create_aliases.py \
    --data $BOOTLEG_PREP_WIKIDATA_DIR \
    --out_file $BOOTLEG_PREP_WIKIPEDIA_DIR/augmented_alias_map_large.jsonl \
    $STRIP_P $LOWER_P

