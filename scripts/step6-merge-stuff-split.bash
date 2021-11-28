echo
echo "=============================================================================="
echo "Step step6-merge-stuff-split"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/merge_shuff_split.py \
    --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
    --subfolder_name full_wiki \
    --split 10
