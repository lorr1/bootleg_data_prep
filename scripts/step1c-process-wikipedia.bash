echo
echo "=============================================================================="
echo "Step step1c-extract-wikipedia"
echo "=============================================================================="
echo
source ./envs.bash
cd $BOOTLEG_PREP_WIKIPEDIA_DIR
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/process_extracted_wikipedia.py \
    --wikiextractor_output wikiextractor_output \
    --output_dir . \
    --processes $BOOTLEG_PREP_PROCESS_COUNT_MIN