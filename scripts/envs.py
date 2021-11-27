from pathlib import Path

from config import *
import pycountry

lang_module_name = None
for lang in pycountry.languages:
    lang_code = getattr(lang, 'alpha_2', lang.alpha_3).lower()
    if lang_code == BOOTLEG_LANG_CODE:
        lang_module_name = lang.name.lower().replace(' ', '-')
if not lang_module_name:
    raise Exception(f'BOOTLEG_LANG_CODE set to wrong language code "{BOOTLEG_LANG_CODE}"')

path_to_code_dir = Path(__file__).resolve().parent.parent

envs = {
    'BOOTLEG_BASE_DIR': f'{BOOTLEG_BASE_DIR}',
    'BOOTLEG_CODE_DIR': f'{path_to_code_dir}',
    'BOOTLEG_LANG_MODULE': f'{lang_module_name}',
    'BOOTLEG_LANG_CODE': f'{BOOTLEG_LANG_CODE}',
    'BOOTLEG_PROCESS_COUNT': f'{BOOTLEG_PROCESS_COUNT}',
    'BOOTLEG_PROCESS_COUNT_IN_LOW_MEMORY_LOAD': f'{BOOTLEG_PROCESS_COUNT_IN_LOW_MEMORY_LOAD}',
    'BOOTLEG_PROCESS_COUNT_IN_HIGH_MEMORY_LOAD': f'{BOOTLEG_PROCESS_COUNT_IN_HIGH_MEMORY_LOAD}',
    'BOOTLEG_WIKIDATA_DIR': f'{BOOTLEG_BASE_DIR}/wikidata_{BOOTLEG_LANG_CODE}',
    'BOOTLEG_WIKIPEDIA_DIR': f'{BOOTLEG_BASE_DIR}/{BOOTLEG_LANG_CODE}_data',
    'BOOTLEG_WIKIPEDIA_DUMP_URL': f'https://dumps.wikimedia.org/{BOOTLEG_LANG_CODE}wiki/latest/{BOOTLEG_LANG_CODE}wiki-latest-pages-articles-multistream.xml.bz2',
    'BOOTLEG_WIKIPEDIA_DUMP_BZ2_FILENAME': f'{BOOTLEG_LANG_CODE}wiki-latest-pages-articles-multistream.xml.bz2',
    'BOOTLEG_WIKIPEDIA_DUMP_FILENAME': f'{BOOTLEG_LANG_CODE}wiki-latest-pages-articles-multistream.xml',
    'BOOTLEG_OUTPUT_DIR': f'{BOOTLEG_BASE_DIR}/output',
    'BOOTLEG_OUTPUT_LOGS_DIR': f'{BOOTLEG_BASE_DIR}/output/logs',
}

if USE_WEAK_LABELING:
    envs['USE_WEAK_LABELING'] = '1'

for key, value in envs.items():
    print(f'export {key}={value}')
