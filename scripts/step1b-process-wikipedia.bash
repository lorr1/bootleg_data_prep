echo
echo "=============================================================================="
echo "Step step1b-process-wikipedia"
echo "=============================================================================="
echo
source ./envs.bash
cd $BOOTLEG_WIKIPEDIA_DIR
python3 -m bootleg_data_prep.wiki_extractor --output sentences --output2 pageids $BOOTLEG_WIKIPEDIA_DUMP_FILENAME --processes $BOOTLEG_PROCESS_COUNT_IN_HIGH_MEMORY_LOAD
