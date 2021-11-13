echo
echo "=============================================================================="
echo "Step step7b-prep-generate-slices-part2"
echo "=============================================================================="
echo
source ./envs.bash
python3 -m bootleg_data_prep.prep_generate_slices_part2 \
  --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
  --subfolder_name full_wiki \
  --processes $BOOTLEG_PROCESS_COUNT \
  --emb_dir $BOOTLEG_WIKIPEDIA_DIR/embs \
  --kg_adj kg_adj_1129.txt \
  --kg_triples kg_triples_1129.json \
  --hy_vocab '' \
  --hy_types '' \
  --wd_vocab wikidatatitle_to_typeid_1129.json \
  --wd_types wikidata_types_1129.json \
  --rel_vocab relation_to_typeid_1129.json \
  --rel_types kg_relation_types_1129.json \
  --overwrite_saved_entities
