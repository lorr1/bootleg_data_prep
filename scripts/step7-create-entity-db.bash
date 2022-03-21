echo
echo "=============================================================================="
echo "Step step7-create-entity-db"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/create_entity_db.py \
  --subfolder_name full_wiki \
  --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
  --ent_desc $BOOTLEG_PREP_WIKIDATA_DIR/wikidata_output/qid2desc.json \
  --kg_triples $BOOTLEG_PREP_WIKIDATA_DIR/wikidata_output/kg_triples.json \
  --kg_vocab $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/utils/param_files/pid_names_$BOOTLEG_PREP_LANG_CODE.json \
  --wd_vocab $BOOTLEG_PREP_WIKIDATA_DIR/wikidata_output/wikidatatitle_to_typeid.json \
  --wd_types $BOOTLEG_PREP_WIKIDATA_DIR/wikidata_output/wikidata_types.json