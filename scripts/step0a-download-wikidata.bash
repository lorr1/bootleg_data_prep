source ./envs.bash
mkdir $BOOTLEG_BASE_DIR/wikidata/
mkdir $BOOTLEG_BASE_DIR/wikidata/raw_data/
cd $BOOTLEG_BASE_DIR/wikidata/raw_data/
aria2c -s 16 -x 16 https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz
bzip -d latest-all.json.gz

