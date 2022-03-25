#cd "$(dirname "${BASH_SOURCE[0]}")"
#source ./setx.bash BOOTLEG_PREP_DATA_DIR "/nvme2/chatterbox/bootleg"
#source ./setx.bash BOOTLEG_PREP_LANG_MODULE hebrew
#source ./setx.bash BOOTLEG_PREP_LANG_CODE he
#source ./setx.bash STRIP_PUNC_ALIASES false
#source ./setx.bash LOWER_CASE_ALIASES true
#source ./setx.bash BOOTLEG_PREP_PROCESS_COUNT_MAX 16
#source ./setx.bash BOOTLEG_PREP_PROCESS_COUNT_AVG 10
#source ./setx.bash BOOTLEG_PREP_PROCESS_COUNT_MIN 6
#source ./setx.bash BOOTLEG_PREP_USE_GPU false
#source ./setx.bash BOOTLEG_PREP_WEAK_LABELING false
#source ./setx.bash BOOTLEG_PREP_PRN_LABELING false

# If Using ZSH
export BOOTLEG_PREP_DATA_DIR="/lfs/raiders8/0/lorr1"
export BOOTLEG_PREP_LANG_MODULE=chinese
export BOOTLEG_PREP_LANG_CODE=zh
export STRIP_PUNC_ALIASES=false
export LOWER_CASE_ALIASES=false
export BOOTLEG_PREP_PROCESS_COUNT_MAX=15
export BOOTLEG_PREP_PROCESS_COUNT_AVG=10
export BOOTLEG_PREP_PROCESS_COUNT_MIN=10
export BOOTLEG_PREP_USE_GPU=false
export BOOTLEG_PREP_WEAK_LABELING=false
export BOOTLEG_PREP_PRN_LABELING=false
