import time
import os
from collections import defaultdict
import bootleg_data_prep.utils.entity_symbols_for_signals as es_for_signals


def load_esp(name, cache_dir="", subfolder="", suffix="", overwrite=False,
             data_dir='/dfs/scratch0/lorr1/data_prep/data/',
             emb_dir='/dfs/scratch0/lorr1/data_prep/embs/',
             max_types=3,
             max_types_rel=50,
             kg_adj='kg_adj_0905.txt',
             kg_triples='kg_triples_0905.txt',
             hy_vocab='hyena_vocab.json',
             hy_types='hyena_types_0905.json',
             wd_vocab='wikidata_to_typeid_0905.json',
             wd_types='wikidata_types_0905.json',
             rel_vocab='relation_to_typeid_0905.json',
             rel_types='kg_relation_types_0905.json'):
    if name not in ["kore", "wiki", "wiki_unfilt", "wiki_0906_pg"]:
        assert len(subfolder) > 0 and len(suffix) > 0, f"{name} is not one of the predefined options of [kore, aida, wiki, wiki_unfilt, wiki_0906_pg]. " \
                                                       f"You must provide an associated subfolder and suffix for the new option. " \
                                                       f"We will look in {data_dir}/{subfolder} for the entity_dump with the required subfolders" \
                                                       f" and save a cache locally under {suffix}."
    if name == "kore":
        subfolder="kore_0929"
        suffix = "_kore"
    elif name == "wiki":
        subfolder="wiki0516"
        suffix = "_wikidata"
    elif name == "wiki_unfilt":
        subfolder="wiki0516_unfilt"
        suffix = "_wikidata_unfilt"
    elif name == "wiki_0906_pg":
        subfolder="wiki_0906_pg"
        suffix = "_wiki_0906_pg"

    st = time.time()
    cache_dir = os.path.join(cache_dir, f"_cache{suffix}")
    es_args = es_for_signals.create_args(data_dir=data_dir,
                    subfolder_name=subfolder,
                    emb_dir=emb_dir,
                    max_types=max_types,
                    max_types_rel=max_types_rel,
                    kg_adj=kg_adj,
                    kg_triples=kg_triples,
                    hy_vocab=hy_vocab,
                    hy_types=hy_types,
                    wd_vocab=wd_vocab,
                    wd_types=wd_types,
                    rel_vocab=rel_vocab,
                    rel_types=rel_types,
                    overwrite_saved_entities=overwrite)
    es, esp, type_vocab_hy, typeid2typename_hy, type_vocab_wd, typeid2typename_wd, type_vocab_rel, typeid2typename_rel, alias_qid_count = es_for_signals.load_entity_symbol_objs(es_args, cache_dir, overwrite_cache=overwrite)
    print(f"Found {len(typeid2typename_hy)} hyena types")
    print(f"Found {len(typeid2typename_wd)} wikidata types")
    print(f"Found {len(typeid2typename_rel)} relation types")
    print(f"FINISHED LOADING IN {time.time() - st}")
    return es, esp, type_vocab_hy, typeid2typename_hy, type_vocab_wd, typeid2typename_wd, type_vocab_rel, typeid2typename_rel, alias_qid_count


def get_qid_types(qid, esp, typeid2typename, type="wikidata"):
    assert type in ["wikidata", "hyena", "relation"], f"You must have a type for get types from wikidata, hyena, or relation. You have {type}"
    if type == "wikidata":
        if esp.qid_in_qid2typeid_wd(qid):
            return tuple([typeid2typename[t] for t in sorted(list(esp.get_types_wd(qid)))])
        else:
            return tuple([])
    elif type == "hyena":
        if esp.qid_in_qid2typeid(qid):
            return tuple([typeid2typename[t] for t in sorted(list(esp.get_types(qid)))])
        else:
            return tuple([])
    else:
        if esp.qid_in_qid2typeid_rel(qid):
            return tuple([typeid2typename[t] for t in sorted(list(esp.get_types_rel(qid)))])
        else:
            return tuple([])


def get_related_qids(qid, esp):
    if esp.qid_in_rel_mapping(qid):
        return tuple(esp.get_relations(qid))
    return tuple([])


def get_all_relationships(qids, esp):
    all_relations = defaultdict(lambda: defaultdict(set))
    for q_i, qid1 in enumerate(qids):
        for q_j, qid2 in enumerate(qids):
            if q_i == q_j:
                continue
            rel = esp.get_relation_name(esp.get_relation(qid1, qid2))
            if rel is not None:
                all_relations[qid1][rel].add(qid2)
                all_relations[qid2][rel].add(qid1)
    return all_relations