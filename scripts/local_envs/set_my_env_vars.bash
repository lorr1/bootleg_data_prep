cd "$(dirname "${BASH_SOURCE[0]}")"
source ./setx.bash BOOTLEG_PREP_DATA_DIR "/nvme2/chatterbox/bootleg"
source ./setx.bash BOOTLEG_PREP_LANG_MODULE hebrew
source ./setx.bash BOOTLEG_PREP_LANG_CODE he
source ./setx.bash BOOTLEG_PREP_PROCESS_COUNT_MAX 16
source ./setx.bash BOOTLEG_PREP_PROCESS_COUNT_AVG 10
source ./setx.bash BOOTLEG_PREP_PROCESS_COUNT_MIN 6
source ./setx.bash BOOTLEG_PREP_USE_GPU false
source ./setx.bash BOOTLEG_PREP_WEAK_LABELING false
