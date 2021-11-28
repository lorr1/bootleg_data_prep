echo
echo "=============================================================================="
echo "Step step0b-process-wikidata"
echo "=============================================================================="
echo
source ./envs.bash
cd $BOOTLEG_PREP_DATA_DIR
python3 $BOOTLEG_PREP_CODE_DIR/simple_wikidata_db/preprocess_dump.py \
    --input_file $BOOTLEG_PREP_DATA_DIR/wikidata/raw_data/latest-all.json \
    --out_dir $BOOTLEG_PREP_WIKIDATA_DIR/processed_batches \
    --total_lines 100000000 \
    --batch_size 1000 \
    --num_processes $BOOTLEG_PROCESS_COUNT_MIN \
    --language_id $BOOTLEG_LANG_CODE
