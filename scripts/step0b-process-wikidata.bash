source ./envs.bash
python3 -m simple_wikidata_db.preprocess_dump \
    --input_file $BOOTLEG_BASE_DIR/wikidata/raw_data/latest-all.json \
    --out_dir $BOOTLEG_WIKIDATA_DIR/processed_batches \
    --total_lines 100000000 \
    --batch_size 1000 \
    --language_id $BOOTLEG_LANG_CODE
