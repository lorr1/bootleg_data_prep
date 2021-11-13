echo
echo "=============================================================================="
echo "Step step7c-generate-slices"
echo "=============================================================================="
echo
source ./envs.bash
python3 -m bootleg_data_prep.generate_slices \
  --subfolder_name full_wiki \
  --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
  --processes $BOOTLEG_PROCESS_COUNT \
  --emb_dir $BOOTLEG_WIKIPEDIA_DIR/embs \
  --ent_desc qid2desc_1129.json \
  --kg_adj kg_adj_1129.txt \
  --kg_triples kg_triples_1129.json \
  --hy_vocab ''\
  --hy_types ''\
  --kg_vocab $BOOTLEG_CODE_DIR/bootleg_data_prep/utils/param_files/pid_names_$BOOTLEG_LANG_CODE.json \
  --wd_vocab wikidatatitle_to_typeid_1129.json \
  --wd_types wikidata_types_1129.json \
  --rel_vocab relation_to_typeid_1129.json \
  --rel_types kg_relation_types_1129.json
