echo
echo "=============================================================================="
echo "Step step0b-process-wikidata"
echo "=============================================================================="
echo
source ./envs.bash
python3 -m simple_wikidata_db.preprocess_dump \
    --input_file $BOOTLEG_BASE_DIR/wikidata/raw_data/latest-all.json \
    --out_dir $BOOTLEG_WIKIDATA_DIR/processed_batches \
    --total_lines 100000000 \
    --batch_size 1000 \
    --num_processes $BOOTLEG_PROCESS_COUNT_IN_HIGH_MEMORY_LOAD \
    --language_id $BOOTLEG_LANG_CODE
