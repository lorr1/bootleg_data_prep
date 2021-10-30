import os
lang_module = os.environ.get('BOOTLEG_LANG_MODULE', 'english')
if lang_module == 'english':
    from langs.english import *
elif lang_module == 'hebrew':
    from langs.hebrew import *
