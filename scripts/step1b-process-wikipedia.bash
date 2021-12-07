echo
echo "=============================================================================="
echo "Step step1b-process-wikipedia"
echo "=============================================================================="
echo
source ./envs.bash
cd $BOOTLEG_PREP_WIKIPEDIA_DIR
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/wiki_extractor.py \
    --output sentences \
    --output2 pageids \
    $BOOTLEG_PREP_WIKIPEDIA_DUMP_FULL_FILENAME \
    --processes $BOOTLEG_PREP_PROCESS_COUNT_MIN
