echo === Setting variables ===
source ./envs.bash

echo === Install Packages ===
sudo apt install aria2

#echo === Creating base folder ===
#mkdir $BOOTLEG_BASE_DIR
#mkdir $BOOTLEG_WIKIPEDIA_DIR
#
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
#gh repo clone git@github.com:neelguha/simple-wikidata-db.git
#
#echo === Installing Python Packages ===
#cd bootleg_data_prep
#pip install -r requirements.txt
#python3 setup.py develop


