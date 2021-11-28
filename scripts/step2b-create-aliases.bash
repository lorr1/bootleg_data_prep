echo
echo "=============================================================================="
echo "Step step2b-create-aliases"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/wikidata/create_aliases.py \
    --data $BOOTLEG_PREP_WIKIDATA_DIR \
    --out_file $BOOTLEG_PREP_WIKIPEDIA_DIR/augmented_alias_map_large.jsonl

