echo
echo "=============================================================================="
echo "Step step7a-prep-generate-slices-part1"
echo "=============================================================================="
echo
source ./envs.bash
python3 -m bootleg_data_prep.prep_generate_slices_part1 \
  --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
  --subfolder_name full_wiki \
  --processes $BOOTLEG_PROCESS_COUNT
