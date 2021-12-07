echo
echo "=============================================================================="
echo "Step step3a-curate-aliases"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/curate_aliases.py \
    --min_frequency 2 \
    --sentence_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/sentences \
    --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
    --wd_aliases $BOOTLEG_PREP_WIKIPEDIA_DIR/augmented_alias_map_large.jsonl \
    --title_to_qid $BOOTLEG_PREP_WIKIPEDIA_DIR/title_mappings/title_to_all_ids.jsonl \
    --processes $BOOTLEG_PREP_PROCESS_COUNT_MAX
