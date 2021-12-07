echo
echo "=============================================================================="
echo "Step step7a-prep-generate-slices-part1"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/prep_generate_slices_part1.py \
  --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
  --subfolder_name full_wiki \
  --processes $BOOTLEG_PREP_PROCESS_COUNT_MAX
