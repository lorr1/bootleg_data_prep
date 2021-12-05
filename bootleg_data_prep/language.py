import os
lang_module = os.environ.get('BOOTLEG_PREP_LANG_MODULE')
if lang_module == 'hebrew':
    from langs.hebrew import *
else:
    from langs.english import *
