echo === Setting variables ===
source ./envs.bash

echo === Install Packages ===
sudo apt install aria2 lbzip2 pv pigz

echo === Creating base folders ===
mkdir $BOOTLEG_BASE_DIR
mkdir $BOOTLEG_WIKIPEDIA_DIR
mkdir $BOOTLEG_WIKIDATA_DIR
mkdir $BOOTLEG_WIKIDATA_DIR/processed_batches
mkdir $BOOTLEG_OUTPUT_DIR
mkdir $BOOTLEG_OUTPUT_LOGS_DIR

# The following are just some helper resources for setup - might want to put this in some readme?
#echo === Installing Github CLI ===
#sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0
#sudo apt-add-repository https://cli.github.com/packages
#sudo apt update
#sudo apt install gh
#
#echo === Cloning Projects ===
#gh auth login
#cd $BOOTLEG_BASE_DIR
#gh repo clone https://github.com/lorr1/bootleg_data_prep.git
#gh repo clone git@github.com:neelguha/simple_wikidata_db.git
#
#echo === Installing Python Packages ===
#cd bootleg_data_prep
#pip install -r requirements.txt
#
#echo === Building Source Packages ===
#
#python3 setup.py develop
