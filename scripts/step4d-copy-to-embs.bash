source ./envs.bash
mkdir $BOOTLEG_WIKIPEDIA_DIR/embs
cp $BOOTLEG_WIKIDATA_DIR/wikidata_output/kg_triples.json $BOOTLEG_WIKIPEDIA_DIR/embs/kg_triples_1129.json
cp $BOOTLEG_WIKIDATA_DIR/wikidata_output/kg_adj.txt $BOOTLEG_WIKIPEDIA_DIR/embs/kg_adj_1129.txt
cp $BOOTLEG_WIKIDATA_DIR/wikidata_output/wikidatatitle_to_typeqid.json $BOOTLEG_WIKIPEDIA_DIR/embs/wikidatatitle_to_typeqid_1129.json
cp $BOOTLEG_WIKIDATA_DIR/wikidata_output/wikidatatitle_to_typeid.json $BOOTLEG_WIKIPEDIA_DIR/embs/wikidatatitle_to_typeid_1129.json
cp $BOOTLEG_WIKIDATA_DIR/wikidata_output/wikidata_types.json $BOOTLEG_WIKIPEDIA_DIR/embs/wikidata_types_1129.json
echo "Copied!"