import psutil

PHYSICAL_MEMORY = psutil.virtual_memory().total  # total physical memory (yes that's physical not virtual)
ESTIMATED_LOW_PROCESS_MEMORY_LOAD = 4000000000  # 2GB
ESTIMATED_AVG_PROCESS_MEMORY_LOAD = 10000000000  # 10GB
ESTIMATED_HIGH_PROCESS_MEMORY_LOAD = 24000000000  # 24GB
BOOTLEG_PROCESS_COUNT = min(psutil.cpu_count(), PHYSICAL_MEMORY // ESTIMATED_AVG_PROCESS_MEMORY_LOAD)
BOOTLEG_PROCESS_COUNT_IN_LOW_MEMORY_LOAD = min(psutil.cpu_count(), PHYSICAL_MEMORY // ESTIMATED_LOW_PROCESS_MEMORY_LOAD)
BOOTLEG_PROCESS_COUNT_IN_HIGH_MEMORY_LOAD = min(PHYSICAL_MEMORY // ESTIMATED_HIGH_PROCESS_MEMORY_LOAD, BOOTLEG_PROCESS_COUNT)
BOOTLEG_BASE_DIR = '/nvme2/chatterbox/bootleg'
BOOTLEG_LANG_CODE = 'he'
BOOTLEG_LANG_MODULE_USE_GPU = False
USE_WEAK_LABELING = False

