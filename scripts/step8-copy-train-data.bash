echo
echo "=============================================================================="
echo "Step step8-copy-train-data"
echo "=============================================================================="
echo
source ./envs.bash
mkdir -p $BOOTLEG_PREP_OUTPUT_DIR
cp -r $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/full_wiki/entity_db_final $BOOTLEG_PREP_OUTPUT_DIR/entity_db
cp $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/full_wiki/qid2cnt.json $BOOTLEG_PREP_OUTPUT_DIR/qid2cnt.json
cat $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/full_wiki/train/out_*jsonl > $BOOTLEG_PREP_OUTPUT_DIR/train.jsonl
cat $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/full_wiki/test/out_*jsonl > $BOOTLEG_PREP_OUTPUT_DIR/test.jsonl
cat $BOOTLEG_PREP_WIKIPEDIA_DIR/data/wiki_dump/full_wiki/dev/out_*jsonl > $BOOTLEG_PREP_OUTPUT_DIR/dev.jsonl
echo "Copied! to $BOOTLEG_PREP_OUTPUT_DIR"
