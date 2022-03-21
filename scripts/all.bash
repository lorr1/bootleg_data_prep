source ./envs.bash
mkdir -p $BOOTLEG_PREP_OUTPUT_DIR
mkdir -p $BOOTLEG_PREP_OUTPUT_LOGS_DIR
#./setup.bash
./step0a-download-wikidata.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step0a-download-wikidata.log
./step0b-process-wikidata.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step0b-process-wikidata.log
./step1a-download-wikipedia.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step1a-download-wikipedia.log
./step1b-process-wikipedia.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step1b-process-wikipedia.log
./step2a-map-titles-to-qids.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step2a-map-titles-to-qids.log
./step2b-create-aliases.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step2b-create-aliases.log
./step3a-curate-aliases.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step3a-curate-aliases.log
./step3b-remove-bad-aliases.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step3b-remove-bad-aliases.log
./step4a-get-all-wikipedia-triples.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4a-get-all-wikipedia-triples.log
./step4b-get-types.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4b-get-types.log
./step4c-get-descriptions.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4c-get-descriptions.log
if [ "$BOOTLEG_PREP_WEAK_LABELING" = true ] ; then
  ./step4d-weak-labeling.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4d-weak-labeling.log
fi
if [ "$BOOTLEG_PREP_PRN_LABELING" = true ] ; then
  ./step4e-prnoun-labeling.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step4e-prnoun-labeling.log
fi
./step5-data-filter.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step5-data-filter.log
./step6-merge-stuff-split.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step6-merge-stuff-split.log
./step7-create-entity-db.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step7-create-entity-db.log
./step8-copy-train-data.bash 2>&1 | tee $BOOTLEG_PREP_OUTPUT_LOGS_DIR/step8-copy-train-data.log
