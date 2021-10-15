source ./envs.bash
cd $BOOTLEG_WIKIPEDIA_DIR
python3 -m bootleg_data_prep.wikiextractor.WikiExtractor --output sentences --output2 pageids $BOOTLEG_WIKIPEDIA_DUMP_FILENAME --processes $BOOTLEG_PROCESS_COUNT
