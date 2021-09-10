source ./envs.bash
cd $BOOTLEG_BASE_DIR/simple-wikidata-db/
python3 preprocess_dump.py \
    --input_file $BOOTLEG_BASE_DIR/wikidata/raw_data/latest-all.json \
    --out_dir $BOOTLEG_WIKIPEDIA_DIR/processed_batches \
    --total_lines 90000000 \
    --batch_size 1000 \
    --language_id $BOOTLEG_LANG_CODE
