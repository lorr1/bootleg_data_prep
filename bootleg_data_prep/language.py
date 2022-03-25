import os
lang_module = os.environ.get('BOOTLEG_PREP_LANG_MODULE')
if lang_module == 'hebrew':
    from langs.hebrew import *
elif lang_module == 'english':
    from langs.english import *
elif lang_module == 'chinese':
    from langs.chinese import *
else:
    raise ValueError("BOOTLEG_PREP_LANG_MODULE must be set. See scripts/local_envs/set_my_envs_vars.bash")
