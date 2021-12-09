echo
echo "=============================================================================="
echo "Step step7c-generate-slices"
echo "=============================================================================="
echo
source ./envs.bash
python3 $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/generate_slices.py \
  --subfolder_name full_wiki \
  --data_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump \
  --processes $BOOTLEG_PREP_PROCESS_COUNT_MAX \
  --emb_dir $BOOTLEG_PREP_WIKIPEDIA_DIR/embs \
  --ent_desc qid2desc_092921.json \
  --kg_adj kg_adj_092921.txt \
  --kg_triples kg_triples_092921.json \
  --hy_vocab ''\
  --hy_types ''\
  --kg_vocab $BOOTLEG_PREP_CODE_DIR/bootleg_data_prep/utils/param_files/pid_names_$BOOTLEG_LANG_CODE.json \
  --wd_vocab wikidatatitle_to_typeid_092921.json \
  --wd_types wikidata_types_092921.json \
  --rel_vocab relation_to_typeid_092921.json \
  --rel_types kg_relation_types_092921.json
