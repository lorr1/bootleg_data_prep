echo
echo "=============================================================================="
echo "Step step4g-weak-labeling"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/weak_label_data.py \
    --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
    --filtered_alias_subdir alias_filtered_sentences \
    --out_subdir orig_wl \
    --wd_aliases $BOOTLEG_PREP_WIKIPEDIA_DIR/augmented_alias_map_large.jsonl \
    --processes 60 #$BOOTLEG_PREP_PROCESS_COUNT_MAX \
#    --overwrite