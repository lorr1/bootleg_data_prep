echo === Removing training outputs, logs and temp files ===
source ./envs.bash
rm -rf "${BOOTLEG_PREP_OUTPUT_DIR:?}"/full_wiki_0_-1_final
rm -rf "${BOOTLEG_PREP_OUTPUT_LOGS_DIR:?}"/*
rm -rf "${BOOTLEG_PREP_WIKIDATA_DIR:?}"/processed_batches/*
rm -rf "${BOOTLEG_PREP_WIKIDATA_DIR:?}"/wikidata_output/*
rm -rf "${BOOTLEG_PREP_WIKIPEDIA_DIR:?}"/data
rm -rf "${BOOTLEG_PREP_WIKIPEDIA_DIR:?}"/embs
rm -rf "${BOOTLEG_PREP_WIKIPEDIA_DIR:?}"/pageids
rm -rf "${BOOTLEG_PREP_WIKIPEDIA_DIR:?}"/sentences
rm -rf "${BOOTLEG_PREP_WIKIPEDIA_DIR:?}"/title_mappings

echo === Cleaned ===