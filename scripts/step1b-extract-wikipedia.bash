echo
echo "=============================================================================="
echo "Step step1b-extract-wikipedia"
echo "=============================================================================="
echo
source ./envs.bash
cd $BOOTLEG_PREP_WIKIPEDIA_DIR
python3 -m wikiextractor.WikiExtractor $BOOTLEG_PREP_WIKIPEDIA_DUMP_FULL_FILENAME \
       --json \
       --processes $BOOTLEG_PREP_PROCESS_COUNT_MIN \
       --output wikiextractor_output \
       --bytes 100M \
       --templates wikiextractor_templates.txt \
       --links