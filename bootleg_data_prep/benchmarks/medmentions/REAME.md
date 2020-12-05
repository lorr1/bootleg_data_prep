UMLS concepts come from
`wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/umls_2017_aa_cat0129.json`

If you want to extract manually, you must first download UMLS (2017AA)

Download
1. download the HTML page to get an "execution" id. add the target file in the service param
wget --save-cookies cookies.txt      --keep-session-cookies       https://utslogin.nlm.nih.gov/cas/login?service=https://download.nlm.nih.gov/umls/kss/2017AA/umls-2017AA-active.zip
2. look at the downloaded page and copy out the execution input value (it's a hidden input of a form)
less login\?service\=https\:%2F%2Fdownload.nlm.nih.gov%2Fumls%2Fkss%2F2017AA%2Fumls-2017AA-active.zip
3. download it. Replace the actual values of execution and password
wget --save-cookies cookies.txt      --keep-session-cookies      --post-data "_eventId=submit&username=xiaoling&password=$PASSWORD&execution=$EXECUTION" https://utslogin.nlm.nih.gov/cas/login?service=https://download.nlm.nih.gov/umls/kss/2017AA/umls-2017AA-active.zip
4. rename the downloaded file


Install X11 on mac: add 
    ForwardX11 yes
    XAuthLocation /opt/X11/bin/xauth
to ~/.ssh/config

On server, you may need to 
1. install xauth
2. conda install unzip lib
3. change the unzip in run_linux.sh to "/home/xling/.conda/envs/xling/bin/unzip" 
4. conda install xorg-libxtst -c conda-forge
5. LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xling/.conda/envs/xling/lib/ ./run_linux.sh


To extract the aliases, install scispacy https://github.com/allenai/scispacy

Then, run
```
from scispacy import umls_utils
import json

d = {}
umls_utils.read_umls_concepts("/dfs/scratch0/lorr1/UMLS/data/2017AA/META", d)
umls_utils.read_umls_types("/dfs/scratch0/lorr1/UMLS/data/2017AA/META", d)
umls_utils.read_umls_definitions("/dfs/scratch0/lorr1/UMLS/data/2017AA/META", d)

with open("/dfs/scratch0/lorr1/MedMentions/full/umls_concepts.json", "w") as out_f:
    json.dump(d, out_f)
```
