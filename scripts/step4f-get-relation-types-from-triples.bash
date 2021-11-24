source ./envs.bash
python3 -m bootleg_data_prep.wikidata.get_relation_types_from_triples \
    --pid_file $BOOTLEG_CODE_DIR/bootleg_data_prep/utils/param_files/pid_names_$BOOTLEG_LANG_CODE.json \
    --kg_triples $BOOTLEG_WIKIPEDIA_DIR/embs/kg_triples_1129.json \
    --type_file $BOOTLEG_WIKIPEDIA_DIR/embs/wikidata_types_1129.json \
    --output_file $BOOTLEG_WIKIPEDIA_DIR/embs/kg_relation_types_1129.json \
    --output_vocab_file $BOOTLEG_WIKIPEDIA_DIR/embs/relation_to_typeid_1129.json

