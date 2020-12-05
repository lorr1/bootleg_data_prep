ALIAS2QID = "alias2qid"
# _hy is for hyena types
QID2TYPEID_HY = "qid2typeid_hy"
# _wd is for wikidata types
QID2TYPEID_WD = "qid2typeid_wd"
# _rel is for wikidata relations (treated as types)
QID2TYPEID_REL = "qid2typeid_rel"
RELMAPPING = "rel_mapping"
ENTITYWORDS = "bag_of_words"
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

CONTEXTUAL_RELATIONS = ["P26"]
POSSIBLE_SLICE_FOLDERS = ["train_prepped", "dev_prepped", "test_prepped"]
PAIR_IDX_MATTER = ["K_most_popular", "K_least_popular", "tail_qids", "bottom_half", "hard_to_disambig", "skewed_disambig"]

# BENCHMARK CONSTANTS 
PREFIX_DELIM = "[SEP]" # we put this between the AIDA document title and the AIDA sentence