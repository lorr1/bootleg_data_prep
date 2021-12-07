echo
echo "=============================================================================="
echo "Step step4h-prnoun-labeling"
echo "=============================================================================="
echo
source ./envs.bash
echo "THIS IS EXPERIMENTAL AND REQUIRES RUNNING THE BELOW FIRST FOR THE FIRST TIME RUNNING"
echo "python3 -m bootleg_data_prep.wikidata.get_person_gender_qids \
        --data $BOOTLEG_PREP_WIKIDATA_DIR \
        --out_dir wikidata_output \
        --processes 10"
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/prn_labels.py \
    $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/orig_wl \
    $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/orig_wl_prn \
    $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/orig_wl/entity_db/entity_mappings \
    --num_workers $BOOTLEG_PREP_PROCESS_COUNT_MAX \
    --swap_titles