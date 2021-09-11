from functools import wraps

UNK_AL = '_unk_'

ALIAS2QID = "alias2qid"
# _hy is for hyena types
QID2TYPEID_HY = "qid2typeid_hy"
# _wd is for wikidata types
QID2TYPEID_WD = "qid2typeid_wd"
# _rel is for wikidata relations (treated as types)
QID2TYPEID_REL = "qid2typeid_rel"
RELMAPPING = "rel_mapping"
TYPEWORDS = "type_words"
RELATIONWORDS = "relation_words"
VOCAB = "all_words"
VOCABFILE = "all_words_vocab"
QIDCOUNT = "qid_count"
CTXRELS = "contextual_rel"
NOCTX = "NCTX"
NOHEADCAND = "NPOP"
NOSINGLE = "NS"
TAILONLY = "TL"
TOESONLY = "TS"
TORSOONLY = "TO"
HEADONLY = "HD"

POSSIBLE_SLICE_FOLDERS = ["train_prepped", "dev_prepped", "test_prepped"]
PAIR_IDX_MATTER = ["K_most_popular", "K_least_popular", "tail_qids", "bottom_half", "hard_to_disambig", "skewed_disambig"]

# BENCHMARK CONSTANTS 
PREFIX_DELIM = "[SEP]" # we put this between the AIDA document title and the AIDA sentence

def edit_op(func):
    @wraps(func)
    def wrapper_check_edit_mode(obj, *args, **kwargs):
        if obj.edit_mode is False:
            raise AttributeError(f"You must load object in edit_mode=True")
        return func(obj, *args, **kwargs)

    return wrapper_check_edit_mode


def check_qid_exists(func):
    @wraps(func)
    def wrapper_check_qid(obj, *args, **kwargs):
        if len(args) > 0:
            qid = args[0]
        else:
            qid = kwargs["qid"]
        if not obj._entity_symbols.qid_exists(qid):
            raise ValueError(f"The entity {qid} is not in our dump")
        return func(obj, *args, **kwargs)

    return wrapper_check_qid