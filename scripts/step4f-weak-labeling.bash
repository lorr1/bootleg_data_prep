echo
echo "=============================================================================="
echo "Step step4f-weak-labeling"
echo "=============================================================================="
echo
source ./envs.bash
python3 -m bootleg_data_prep.weak_label_data \
    --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
    --filtered_alias_subdir alias_filtered_sentences \
    --out_subdir weak_labeling_data \
    --wd_aliases $BOOTLEG_WIKIPEDIA_DIR/augmented_alias_map_large.jsonl \
    --processes $BOOTLEG_PROCESS_COUNT \
    --overwrite \

