# The following environment variables are required:
# =================================================
# $BOOTLEG_PREP_DATA_DIR - the output/intermediate bootleg data folder
# $BOOTLEG_PREP_LANG_MODULE - the language folder name under /bootleg/langs. for example: "english"
# $BOOTLEG_PREP_LANG_CODE - the language code. for example: en
# $BOOTLEG_PREP_PROCESS_COUNT_MAX - processes to run concurrently, basically: CPU core count
# $BOOTLEG_PREP_PROCESS_COUNT_AVG - processes to run concurrently, based on the amount machine's physical memory divided by 10G
# $BOOTLEG_PREP_PROCESS_COUNT_MIN - processes to run concurrently, based on the amount machine's physical memory divided by 24G
# $BOOTLEG_PREP_USE_GPU - should GPU be used
# $BOOTLEG_PREP_WEAK_LABELING - if "true" weak labeling will be included in the prep process

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
UP_SCRIPT_DIR=$(builtin cd $SCRIPT_DIR/..; pwd)
export BOOTLEG_PREP_CODE_DIR=$UP_SCRIPT_DIR
export PYTHONPATH=$BOOTLEG_PREP_CODE_DIR
export BOOTLEG_PREP_BENCHMARKS_DIR=$BOOTLEG_PREP_CODE_DIR/benchmarks
export BOOTLEG_PREP_CONFIGS_DIR=$PATH_TO_CODE_DIR/configs
export BOOTLEG_PREP_LOGS_DIR=$BOOTLEG_PREP_CODE_DIR/logs
export BOOTLEG_PREP_BERT_CACHE_DIR=$BOOTLEG_PREP_CODE_DIR/bert_cache
export BOOTLEG_PREP_WIKIDATA_DIR=$BOOTLEG_PREP_DATA_DIR/wikidata_"$BOOTLEG_PREP_LANG_CODE"
#export BOOTLEG_PREP_WIKIDATA_DIR=$BOOTLEG_PREP_DATA_DIR/wikidata_09_21
export BOOTLEG_PREP_WIKIPEDIA_DIR=$BOOTLEG_PREP_DATA_DIR/"$BOOTLEG_PREP_LANG_CODE"_data
export BOOTLEG_PREP_WIKIPEDIA_DUMP_URL=https://dumps.wikimedia.org/"$BOOTLEG_PREP_LANG_CODE"wiki/latest/"$BOOTLEG_PREP_LANG_CODE"wiki-latest-pages-articles-multistream.xml.bz2
export BOOTLEG_PREP_WIKIPEDIA_DUMP_BZ2_FILENAME="$BOOTLEG_PREP_LANG_CODE"wiki-latest-pages-articles-multistream.xml.bz2
export BOOTLEG_PREP_WIKIPEDIA_DUMP_FILENAME="$BOOTLEG_PREP_LANG_CODE"wiki-latest-pages-articles-multistream.xml
export BOOTLEG_PREP_WIKIPEDIA_DUMP_FULL_FILENAME="$BOOTLEG_PREP_WIKIPEDIA_DIR"/"$BOOTLEG_PREP_LANG_CODE"wiki-latest-pages-articles-multistream.xml
export BOOTLEG_PREP_OUTPUT_DIR=$BOOTLEG_PREP_DATA_DIR/output
export BOOTLEG_PREP_OUTPUT_LOGS_DIR=$BOOTLEG_PREP_DATA_DIR/output/logs
