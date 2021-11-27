source ./envs.bash
mkdir -p $BOOTLEG_OUTPUT_DIR
mkdir -p $BOOTLEG_OUTPUT_LOGS_DIR
#./setup.bash
#./step0a-download-wikidata.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step0a-download-wikidata.log
./step0b-process-wikidata.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step0b-process-wikidata.log
#./step1a-download-wikipedia.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/.log
./step1b-process-wikipedia.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step1b-process-wikipedia.log
./step2a-map-titles-to-qids.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step2a-map-titles-to-qids.log
./step2b-create-aliases.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step2b-create-aliases.log
./step3a-curate-aliases.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step3a-curate-aliases.log
./step3b-remove-bad-aliases.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step3b-remove-bad-aliases.log
./step4a-get-all-wikipedia-triples.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step4a-get-all-wikipedia-triples.log
./step4b-create-kg-adj.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step4b-create-kg-adj.log
./step4c-get-types.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step4c-get-types.log
./step4d-copy-to-embs.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step4d-copy-to-embs.log
./step4e-get-relation-types-from-triples.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step4e-get-relation-types-from-triples.log
if test ! -z "$USE_WEAK_LABELING"; then
  ./step4f-weak-labeling.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step4f-weak-labeling.log
fi
./step5-data-filter.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step5-data-filter.log
./step6-merge-stuff-split.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step6-merge-stuff-split.log
./step7a-prep-generate-slices-part1.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step7a-prep-generate-slices-part1.log
./step7b-prep-generate-slices-part2.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step7b-prep-generate-slices-part2.log
./step7c-generate-slices.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step7c-generate-slices.log
./step8-copy-train-data.bash 2>&1 | tee $BOOTLEG_OUTPUT_LOGS_DIR/step8-copy-train-data.log
