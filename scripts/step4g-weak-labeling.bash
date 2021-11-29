echo
echo "=============================================================================="
echo "Step step4g-weak-labeling"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/weak_label_data.py \
    --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
    --filtered_alias_subdir alias_filtered_sentences \
    --out_subdir weak_labeling_data \
    --wd_aliases $BOOTLEG_PREP_WIKIPEDIA_DIR/augmented_alias_map_large.jsonl \
    --processes $BOOTLEG_PROCESS_COUNT_MAX \
    --overwrite \

