echo
echo "=============================================================================="
echo "Step step1a-download-wikipedia"
echo "=============================================================================="
echo
source ./envs.bash
cd $BOOTLEG_PREP_WIKIPEDIA_DIR
echo "Downloading from $BOOTLEG_PREP_WIKIPEDIA_DUMP_URL"
#aria2c -s 16 -x 16 $BOOTLEG_PREP_WIKIPEDIA_DUMP_URL
wget --no-check-certificate $BOOTLEG_PREP_WIKIPEDIA_DUMP_URL .
rm $BOOTLEG_PREP_WIKIPEDIA_DUMP_FILENAME
lbzip2 -d $BOOTLEG_PREP_WIKIPEDIA_DUMP_BZ2_FILENAME
