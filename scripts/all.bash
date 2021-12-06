source ./envs.bash
if [ "$1" == "download" ]; then
  mkdir -p $BOOTLEG_PREP_OUTPUT_DIR
  mkdir -p $BOOTLEG_PREP_OUTPUT_LOGS_DIR
  ./setup.bash
  ./step0a-download-wikidata.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step0a-download-wikidata.log
  ./step1a-download-wikipedia.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step1a-download-wikipedia.log
fi
if [ "$1" != "continue" ]; then
  ./delete_all_but_downloads.bash
fi
./step0b-process-wikidata.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step0b-process-wikidata.log
./step1b-process-wikipedia.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step1b-process-wikipedia.log
./step2a-map-titles-to-qids.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step2a-map-titles-to-qids.log
./step2b-create-aliases.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step2b-create-aliases.log
./step3a-curate-aliases.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step3a-curate-aliases.log
./step3b-remove-bad-aliases.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step3b-remove-bad-aliases.log
./step4a-get-all-wikipedia-triples.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4a-get-all-wikipedia-triples.log
./step4b-create-kg-adj.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4b-create-kg-adj.log
./step4c-get-types.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4c-get-types.log
./step4d-get-descriptions.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4d-get-descriptions.log
./step4e-copy-to-embs.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4e-copy-to-embs.log
./step4f-get-relation-types-from-triples.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4f-get-relation-types-from-triples.log
if [ "$BOOTLEG_PREP_WEAK_LABELING" = true ] ; then
  ./step4g-weak-labeling.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4g-weak-labeling.log
fi
if [ "$BOOTLEG_PREP_PRN_LABELING" = true ] ; then
  ./step4h-prnoun-labeling.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4h-prn-labeling.log
fi
./step5-data-filter.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step5-data-filter.log
./step6-merge-stuff-split.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step6-merge-stuff-split.log
./step7a-prep-generate-slices-part1.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step7a-prep-generate-slices-part1.log
./step7b-prep-generate-slices-part2.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step7b-prep-generate-slices-part2.log
./step7c-generate-slices.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step7c-generate-slices.log
./step8-copy-train-data.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step8-copy-train-data.log
