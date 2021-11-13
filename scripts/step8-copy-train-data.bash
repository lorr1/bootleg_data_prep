echo
echo "=============================================================================="
echo "Step step8-copy-train-data"
echo "=============================================================================="
echo
source ./envs.bash
cp -r $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump/full_wiki_0_-1_final $BOOTLEG_OUTPUT_DIR
cp $BOOTLEG_WIKIPEDIA_DIR/embs/kg_adj_1129.txt $BOOTLEG_OUTPUT_DIR/full_wiki_0_-1_final/entity_db/kg_adj_1129.txt
