import argparse
import glob
import multiprocessing
import os
import random
import shutil
import time
from collections import defaultdict
from typing import Any, Set
import numpy as np
import ujson
from tqdm import tqdm

import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.utils.classes.entity_symbols import EntitySymbols
from bootleg_data_prep.utils import utils, weak_label_funcs
from bootleg_data_prep.utils.classes.record_trie_collection import RecordTrieCollection
from bootleg_data_prep.utils.weak_label_funcs import wl_func

ALIAS2QID = "alias2qids"
QID2ALIAS = "qid2alias"

class WLMetadata:
    def __init__(self, entity_dump=None, qid2title=None, tri_collection_qids=None, tri_collection_aliases=None):
        if entity_dump is not None:
            print(f"entity_dump is not None. Rebuilding WLMetadata")
            alias2qids = {}
            qid2alias = {}
            self.qid2title = entity_dump.get_qid2title()
            all_alias_len = []
            all_qid_len = []
            for alias in tqdm(entity_dump.get_all_aliases(), desc="Iterating over aliases"):
                assert len(alias.strip()) > 0
                alias2qids[alias] = []
                cands = entity_dump.get_qid_cands(alias)
                for idx, qid in enumerate(cands):
                    if not qid in qid2alias:
                        qid2alias[qid] = []
                    qid2alias[qid].append(alias)
                    alias2qids[alias].append(qid)
                all_qid_len.append(len(alias2qids[alias]))

            for qid, alias_cands in tqdm(qid2alias.items(), desc="Iterating over qids"):
                all_alias_len.append(len(alias_cands))

            max_cands = 100
            max_aliases = 100
            print(f"Average number of connections {np.average(all_qid_len)}, 99.9th percentile {np.percentile(all_qid_len, 99.9)} - Trimming to {max_cands}")
            print(f"Average number of connections {np.average(all_alias_len)}, 99.9th percentile {np.percentile(all_alias_len, 99.9)} - Trimming to {max_aliases}")
            for alias in tqdm(list(alias2qids.keys()), desc="Iterating over aliases"):
                alias2qids[alias] = alias2qids[alias][:max_cands]

            for qid in tqdm(list(qid2alias.keys()), desc="Iterating over qids"):
                qid2alias[qid] = qid2alias[qid][:max_aliases]

            # This maps our keys that we use in the helper functions below to the right tri in tri collection.
            # The values are specific strings as outlines in the record trie collection class
            fmt_types = {ALIAS2QID: "qid_cand"}
            max_values = {ALIAS2QID: max_cands}
            input_dicts = {ALIAS2QID: alias2qids}

            print(f"Max Values {max_values}")
            self.tri_collection_qids = RecordTrieCollection(load_dir=None, input_dicts=input_dicts, vocabulary=entity_dump.get_qid2eid(),
                                                       fmt_types=fmt_types, max_values=max_values)

            # This maps our keys that we use in the helper functions below to the right tri in tri collection.
            # The values are specific strings as outlines in the record trie collection class
            fmt_types = {QID2ALIAS: "qid_cand"}
            max_values = {QID2ALIAS: max_aliases}
            input_dicts = {QID2ALIAS: qid2alias}
            alias_vocab = {al:i for i, al in enumerate(alias2qids.keys())}

            print(f"Max Values {max_values}")
            self.tri_collection_aliases = RecordTrieCollection(load_dir=None, input_dicts=input_dicts, vocabulary=alias_vocab,
                                                       fmt_types=fmt_types, max_values=max_values)
        else:
            assert qid2title is not None, f"You have a None entity_dump, we require qid2title to not be None"
            assert tri_collection_qids is not None, f"You have a None entity_dump, we require tri_collection_qids to not be None"
            assert tri_collection_aliases is not None, f"You have a None entity_dump, we require tri_collection_aliases to not be None"
            self.qid2title = qid2title
            self.tri_collection_qids = tri_collection_qids
            self.tri_collection_aliases = tri_collection_aliases

    @classmethod
    def get_qid_tri_dir(cls, dump_dir):
        return os.path.join(dump_dir, "QIDTRI")

    @classmethod
    def get_alias_tri_dir(cls, dump_dir):
        return os.path.join(dump_dir, "ALIASTRI")

    @classmethod
    def get_qid2title_file(cls, dump_dir):
        return os.path.join(dump_dir, "QID2TITLE.json")

    def dump(self, dump_dir):
        self.tri_collection_qids.dump(save_dir=self.get_qid_tri_dir(dump_dir))
        self.tri_collection_aliases.dump(save_dir=self.get_alias_tri_dir(dump_dir))
        with open(self.get_qid2title_file(dump_dir), "w") as out_f:
            ujson.dump(self.qid2title, out_f)

    @classmethod
    def load(cls, dump_dir):
        tri_collection_qids = RecordTrieCollection(load_dir=cls.get_qid_tri_dir(dump_dir))
        tri_collection_aliases = RecordTrieCollection(load_dir=cls.get_alias_tri_dir(dump_dir))
        with open(cls.get_qid2title_file(dump_dir)) as in_f:
            qid2title = ujson.load(in_f)
        return cls(entity_dump=None, qid2title=qid2title, tri_collection_qids=tri_collection_qids, tri_collection_aliases=tri_collection_aliases)

    def contains_qid(self, qid):
        return self.tri_collection_aliases.is_key_in_trie(QID2ALIAS, qid)

    def contains_alias(self, alias):
        return self.tri_collection_qids.is_key_in_trie(ALIAS2QID, alias)

    def get_all_aliases(self, qid: str, default: Any = None) -> Set[str]:
        if self.contains_qid(qid):
            return self.tri_collection_aliases.get_value(QID2ALIAS, qid)
        else:
            return default

    def get_num_cands(self, alias):
        assert self.contains_alias(alias), f"{alias} not in mapping"
        return len(self.tri_collection_qids.get_value(ALIAS2QID, alias))

    def get_cand_pos(self, alias, qid):
        assert self.contains_alias(alias), f"{alias} not in mapping"
        try:
            return self.tri_collection_qids.get_value(ALIAS2QID, alias).index(qid)
        except:
            return -1

    def get_title(self, qid):
        return self.qid2title.get(qid, None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/wiki_dump', help='Directory for data to be saved.')
    parser.add_argument('--filtered_alias_subdir', type=str, default='alias_filtered_sentences', help = 'Subdirectory to save filtered sentences.')
    parser.add_argument('--out_subdir', type = str, default = 'test_wl2', help = 'Where to write processed data to.')
    parser.add_argument('--no_permute_alias', action='store_true', help = 'Turn on to not make the new alias of added entities be the most conflicting.')
    parser.add_argument('--max_candidates', type=int, default=30)
    parser.add_argument('--processes', type=int, default=int(0.1*multiprocessing.cpu_count()))
    parser.add_argument('--overwrite', action = 'store_true', help = 'Rebuild WL metadata.')
    parser.add_argument('--test', action = 'store_true', help = 'If set, will only generate for one file.')

    args = parser.parse_args()
    return args

def init_process(wl_metadata_dump):
    global wl_metadata_global
    wl_metadata_global = WLMetadata.load(wl_metadata_dump)

def launch_subprocess(args, outdir, temp_outdir, wl_metadata_dump, in_files):

    all_process_args = [tuple([i+1,
                               len(in_files),
                               outdir,
                               temp_outdir,
                               args,
                               in_files[i],
                               ]) for i in range(len(in_files))]
    print("Starting pool...")
    pool = multiprocessing.Pool(processes=args.processes, initializer=init_process, initargs=([wl_metadata_dump]))
    print("Starting processes...")
    docs_not_qid = set()
    for docs_not_qid_subset in tqdm(pool.imap(subprocess, all_process_args, chunksize=1), total=len(all_process_args)):
        docs_not_qid.update(set(docs_not_qid_subset))
    pool.close()
    return docs_not_qid

def choose_new_alias(max_cands, alias, qid, wl_metadata, doc_ent, sentence_idx):
    # Set a seed to ensure that across ablations, the aliases chosen will be consistent. For example, if we are processing
    # the document for "Q123" and are on sentence 55 of that article, and are currently labeling the QID "Q88" then we will
    # set the seed to 1235588.
    seed = int(str(doc_ent[1:]) + str(sentence_idx) + str(qid[1:]))
    random.seed(seed)
    if not wl_metadata.contains_qid(qid):
        return alias
    # If qid is in the top 30 for the alias, and there are at least 2 candidates for that alias, just use that alias
    if 0 <= wl_metadata.get_cand_pos(alias, qid) < max_cands and wl_metadata.get_num_cands(alias) > 1:
        return alias
    # Otherwise, find all other aliases for that qid that are in the top 30 and have at least 2 candidates, and randomly choose on
    top_mc_aliases = [al for al in wl_metadata.get_all_aliases(qid) if 0 <= wl_metadata.get_cand_pos(al, qid) < max_cands]
    top_mc_gtr1_cand_aliases = [al for al in top_mc_aliases if wl_metadata.get_num_cands(al) > 1]
    if len(top_mc_gtr1_cand_aliases) > 0:
        return random.choice(top_mc_gtr1_cand_aliases)
    # We might be in the situation where there are a bunch of aliases for that qid (and the top max cands (mc) condition is met) but they
    # all have only 1 candidate. That's better than nothing, so in that case, randomly return one of those aliases.
    if len(top_mc_aliases) > 0:
        return random.choice(top_mc_aliases)
    # If all of the above fail, then just return the original alias
    return alias

def sort_aliases(spans, qids, aliases, sources):
    if len(aliases) == 0:
        return spans, qids, aliases, sources
    res = sorted(zip(spans, qids, aliases, sources), key=lambda x: [x[0][0], x[0][1]], reverse=False)
    spans, qids, aliases, sources = zip(*res)
    return spans, qids, aliases, sources

def subprocess(all_args):
    prep_utils.print_memory()
    start_time = time.time()
    random.seed(1234)

    lfs = list(wl_func.all.values())
    print("LFS", lfs)

    idx, total, outdir, temp_outdir, args, in_filepath = all_args
    num_lines = sum(1 for _ in open(in_filepath))

    # create output files
    out_fname = os.path.join(outdir, prep_utils.get_outfname(in_filepath))

    print(f"Starting {idx}/{total}. Reading in {in_filepath}. Ouputting to {out_fname}")

    filtered_qid_counts = defaultdict(lambda: defaultdict(int))
    filtered_aliases_to_qid_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    no_qid = []
    added_alias = defaultdict(int)
    with open(in_filepath, 'r') as in_file, open(out_fname, "w") as out_file:
        for doc_idx, doc in tqdm(enumerate(in_file), total=num_lines, desc=f"Processing"):
            doc = ujson.loads(doc)

            title = doc['title']
            doc_entity = str(doc['qid'])

            if doc_entity != "-1" and wl_metadata_global.contains_qid(doc_entity):
                no_qid.append(title)

            # Gather all aliases -> qids in the document and qid -> list of aliases
            aliases_to_qids_in_doc, qid_to_aliases_in_doc = collect_aliases_to_qids_in_doc(doc, wl_metadata_global)

            new_sentences = []
            for sentence_idx, line in enumerate(doc['sentences']):
                if len(line['spans']) > 0 and type(line['spans'][0]) is str:
                    line['spans'] = [list(map(int, s.split(":"))) for s in line["spans"]]

                orig_spans, orig_qids, orig_aliases, orig_sources = line["spans"], line["qids"], line["aliases"], ["gold"]*len(line["aliases"])
                added_alias["gold"] += len(orig_aliases)

                for lf in lfs:
                    assert lf.__name__ != "gold", f"We use the name \"gold\" already for a LF. Please name it something else."
                    st = time.time()
                    new_spans, new_qids, new_aliases = lf(doc_entity, line["sentence"], orig_spans,
                                                          orig_qids, orig_aliases,
                                                          aliases_to_qids_in_doc, wl_metadata_global)
                    new_sources = [lf.__name__]*len(new_aliases)
                    assert len(new_spans) == len(new_qids) == len(new_aliases)
                    added_alias[lf.__name__] += len(new_aliases)
                    # print("SENT:", line["sentence"])
                    # for sp, q, al in zip(new_spans, new_qids, new_aliases):
                    #     print("SP:", sp, "AL", al, "Q", q, "LF", lf.__name__)
                    # print(f"Time for lf {lf.__name__} is {time.time()-st}")
                    orig_spans.extend(new_spans)
                    orig_qids.extend(new_qids)
                    orig_aliases.extend(new_aliases)
                    orig_sources.extend(new_sources)

                    orig_spans, orig_qids, orig_aliases, orig_sources = sort_aliases(orig_spans, orig_qids, orig_aliases, orig_sources)

                final_spans, final_qids, final_aliases, final_sources = list(orig_spans), list(orig_qids), list(orig_aliases), list(orig_sources)
                # Permute aliases if flag is turned on
                # If not permuting alias, just use the aliases given. HOWEVER, note that if the qid is not in the top-30 for this alias,
                # then this label will be DROPPED from the training set later on. So it is likely recommended to leave permuting
                # alias ON to prevent this loss of information
                # st = time.time()
                if not args.no_permute_alias:
                    for j in range(len(final_aliases)):
                        alias = final_aliases[j]
                        associated_qid = final_qids[j]
                        new_alias = choose_new_alias(args.max_candidates, alias, associated_qid, wl_metadata_global, doc_entity, line['doc_sent_idx'])
                        final_aliases[j] = new_alias
                # print(f"Time for swap {time.time() - st}")

                new_sentences.append({
                    'doc_sent_idx': line['doc_sent_idx'],
                    'sentence': line['sentence'],
                    'aliases': final_aliases,
                    'spans': final_spans,
                    'qids': final_qids,
                    'gold': [True if fs == "gold" else False for fs in final_sources],
                    'sources': final_sources
                })
                # Update stats
                if len(final_aliases) > 0:
                    for alias, qid, source in zip(final_aliases, final_qids, final_sources):
                        filtered_qid_counts[source][qid] += 1
                        filtered_aliases_to_qid_count[source][alias][qid] += 1
            doc['sentences'] = new_sentences
            out_file.write(ujson.dumps(doc) + '\n')
    out_file.close()
    print(f"LEN: {len(filtered_qid_counts)} LEN: {len(filtered_aliases_to_qid_count)}")
    utils.dump_json_file(os.path.join(temp_outdir, f"filtered_alias_to_qid_count_{idx}.json"), filtered_aliases_to_qid_count)
    utils.dump_json_file(os.path.join(temp_outdir, f"filtered_qid_counts_{idx}.json"), filtered_qid_counts)
    print(f"Finished {idx}/{total}. Written to {out_fname}. {time.time() - start_time} seconds.")
    print(ujson.dumps(added_alias, indent=4))
    return no_qid

def collect_aliases_to_qids_in_doc(doc, wl_metadata):
    """
    :param doc:
    :param wl_metadata:
    :return: aliases_to_qids_in_doc_pruned, qid_to_aliases_in_doc_pruned

    The method gathers a dict of alias->qid->count for all qids linked to in a given document, including the aliases that refer to the QID of the Wikipedia page itself.
    These are then pruned to remove aliases that have different qids that appear with similar frequencies (a sign of a noisy alias).
    """
    st = time.time()
    aliases_to_qids_in_doc = defaultdict(lambda: defaultdict(int))
    doc_entity = str(doc['qid'])

    # Add aliases pointing to the document
    aliases = wl_metadata.get_all_aliases(doc_entity, set())
    for al in aliases:
        assert len(al) > 0
        # We correct this count below when pruning
        aliases_to_qids_in_doc[al][doc_entity] = 1
    # We always add other aliases so when we prune, we can remove highly conflicting aliases to use during weak labelling
    for sentence in doc['sentences']:
        sentence_qids = sentence['qids']
        # Update the aliases_to_qids_in_doc
        for qid in sentence_qids:
            for al in wl_metadata.get_all_aliases(qid, set()):
                aliases_to_qids_in_doc[al][qid] += 1
    aliases_to_qids_in_doc_pruned, qid_to_aliases_in_doc_pruned = prune_aliases_to_qids_in_doc(doc_entity, aliases_to_qids_in_doc)
    # print(f"Time for collect aliases", time.time() - st)
    return aliases_to_qids_in_doc_pruned, qid_to_aliases_in_doc_pruned

def prune_aliases_to_qids_in_doc(doc_entity, aliases_to_qids_in_doc):
    """doc_entity: QID of page we are on
       aliases_to_qids_in_doc: list of aliases on page -> QID they link to UNION all aliases that point to doc_entity -> doc_entity"""
    st = time.time()
    aliases_to_qids_in_doc_pruned = {}
    qid_to_aliases_in_doc_pruned = defaultdict(list)
    total_qid_count = sum(v for qid_dict in aliases_to_qids_in_doc.values() for v in qid_dict.values())
    # print(f"Total Count for {doc_entity} is {total_qid_count}")
    # We want to assign some weight of doc_entity aliases -> doc_entity (they are often not actual links in that Wikipedia so get a low weight by default)
    doc_entity_perc_of_total = 0.2
    # This is representing the popularity of the doc entity in a document. If there is some other QID that appears with some alias
    # also associated with the doc_entity more than doc_entity_count times, we remove this highly conflicting alias from consideration
    popularity_threshold = 5
    doc_entity_count = max(doc_entity_perc_of_total*total_qid_count, 1.0)
    for al in aliases_to_qids_in_doc:
        qid_dict = aliases_to_qids_in_doc[al]
        # qid_dict is qid -> count of number of times al linked to qid; if only one qid, add it for that alias
        # otherwise, find the most popular qid if one exists
        if len(qid_dict) == 1:
            qid_to_add = next(iter(qid_dict.keys()))
            # Add the qid to the list of aliases
            aliases_to_qids_in_doc_pruned[al] = qid_to_add
            qid_to_aliases_in_doc_pruned[qid_to_add].append(al)
        else:
            # Assign the doc entity weight
            if doc_entity in qid_dict:
                qid_dict[doc_entity] = doc_entity_count
            sorted_qids = list(sorted(qid_dict.items(), key=lambda x: x[1], reverse=True))
            # Only add if count is above threshold
            if sorted_qids[0][1] > popularity_threshold*sorted_qids[1][1]:
                qid_to_add = sorted_qids[0][0]
                aliases_to_qids_in_doc_pruned[al] = qid_to_add
                qid_to_aliases_in_doc_pruned[qid_to_add].append(al)
    # print(f"Time for prune aliases", time.time() - st)
    return aliases_to_qids_in_doc_pruned, qid_to_aliases_in_doc_pruned


# Here, we just copy the entity dump over to the new directory but delete erroneous -1 qids
# that crop up if we don't have the page title in our mapping.
def modify_counts_and_dump(args, entity_dump):
    alias2qids = entity_dump.get_alias2qids()
    qid2title = entity_dump.get_qid2title()
    if "-1" in qid2title:
        del qid2title["-1"]

    max_candidates = entity_dump.max_candidates
    max_alias_len = entity_dump.max_alias_len

    for al in alias2qids:
        all_pairs = alias2qids[al]
        qids = [p[0] for p in all_pairs]
        if "-1" in qids:
            print(f"BAD: for alias {al} there is a -1 QID of {alias2qids[al]}. Will remove")
            all_pairs.pop(qids.index("-1"))
            alias2qids[al] = all_pairs

    # Make entity dump object
    entity_dump = EntitySymbols(
        max_candidates=max_candidates,
        max_alias_len=max_alias_len,
        alias2qids=alias2qids,
        qid2title=qid2title
    )
    out_dir = os.path.join(args.data_dir, args.out_subdir, 'entity_db/entity_mappings')
    entity_dump.dump(out_dir)

def main():
    gl_start = time.time()
    multiprocessing.set_start_method("fork", force=True)
    args = parse_args()
    print(ujson.dumps(vars(args), indent=4))
    outdir = prep_utils.get_outdir(args.data_dir, args.out_subdir, remove_old=True)
    temp_outdir = prep_utils.get_outdir(os.path.join(args.data_dir, args.out_subdir), "_temp", remove_old=True)
    temp_metadata_outdir = prep_utils.get_outdir(os.path.join(args.data_dir, args.filtered_alias_subdir), "_for_rerun_WL", remove_old=False)

    # get inputs files 
    path = os.path.join(args.data_dir, args.filtered_alias_subdir, "*.jsonl")
    in_files = prep_utils.glob_files(path)
    # if in test mode, just take a single input file
    if args.test:
        in_files = in_files[:1]

    st = time.time()
    entity_dump = None
    wl_metadata_dump = os.path.join(temp_metadata_outdir, "wl_metadata")
    if not os.path.exists(wl_metadata_dump) or args.overwrite:
        # this loads all entity information (aliases, titles, etc)
        print(f"Reading in entity dump...")
        entity_dump = EntitySymbols(load_dir=os.path.join(args.data_dir, args.filtered_alias_subdir, 'entity_db/entity_mappings'))
        print(f"Loaded entity dump with {entity_dump.num_entities} entities.")

        utils.ensure_dir(wl_metadata_dump)
        wl_metadata = WLMetadata(entity_dump)
        wl_metadata.dump(wl_metadata_dump)
        print(f"Time to create WL metadata {time.time() - st}")

    # launch subprocesses and collect outputs
    print(f"Loaded {len(in_files)} files from {path}. Launching {args.processes} processes.")
    docs_not_qid = launch_subprocess(args, outdir, temp_outdir, wl_metadata_dump, in_files)

    # Gather new counts
    # Total QID count
    qid_count_files = glob.glob(f"{temp_outdir}/filtered_qid_counts_*")
    list_of_qids_dicts = [utils.load_json_file(f) for f in qid_count_files]
    filtered_qid_count = prep_utils.aggregate_list_of_nested_dictionaries(list_of_qids_dicts)
    # Alias, qid pair counts
    aliases_to_qid_count_files = glob.glob(f"{temp_outdir}/filtered_alias_to_qid_count_*")
    list_of_alias_dicts = [utils.load_json_file(f) for f in aliases_to_qid_count_files]
    filtered_aliases_to_qid = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for subdict in list_of_alias_dicts:
        for source_key, subsubdict in subdict.items():
            for alias_key, subsubsubdic in subsubdict.items():
                for qid, cnt in subsubsubdic.items():
                    filtered_aliases_to_qid[source_key][alias_key][qid] += cnt
    # Save counts
    utils.dump_json_file(os.path.join(outdir, "filtered_qid_count.json"), filtered_qid_count)
    utils.dump_json_file(os.path.join(outdir, "filtered_aliases_to_qid_count.json"), filtered_aliases_to_qid)
    with open(os.path.join(outdir, "docs_not_qids.json"), "w") as out_f:
        ujson.dump(docs_not_qid, out_f)

    if entity_dump is None:
        print(f"Reading in entity dump...")
        entity_dump = EntitySymbols(load_dir=os.path.join(args.data_dir, args.filtered_alias_subdir, 'entity_db/entity_mappings'))
        print(f"Loaded entity dump with {entity_dump.num_entities} entities.")

    modify_counts_and_dump(args, entity_dump)
    # remove temp
    shutil.rmtree(temp_outdir)
    vars(args)["out_dir"] = outdir
    prep_utils.save_config(args, "add_labels_single_func_config.json")
    print(f"Finished add_labels_single_func in {time.time() - gl_start} seconds.")

if __name__ == '__main__':
    main()
