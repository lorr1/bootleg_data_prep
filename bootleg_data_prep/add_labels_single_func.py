'''
This file: 
1. Reads data from <data_dir>/alias_filtered_sentences
2. Iterates through each sentence in each document and adds aliases according to predefined 
heuristic functions (more below) -- NOTE, this does not do pronount coref. That is done in a separate process.
3. Writes new data and labels to <data_dir>/orig

The first step of labelling is to collect a mapping for aliases to QIDs for the document. If we find any alias in the document, we label
it with its QID.

To mine for this mapping, we first get all aliases that point to the Wikipedia page we are on. If --no_coref is False, we then
get all aliases for any linked to QID from wikipedia page we are on. This is using the heuristic that if a page points
to some QID, it will likely do so again. This forms our unfiltered alias->qid mapping.

We then filter the alias->qid mapping to remove aliases that are conflicting (i.e., are pointing to different QIDs frequently in the document). We
determine that this alias->qid pair is is too noisy to use.

Once filtered, we use this in our labelling to mine for new aliaeses. If --no_permute_alias is False, we swap all aliases to be the one that is most
conflicting for a QID (largest number of candidates). No matter what no_permute_alias is, if there is an alias with a single candidate, we try
to swap it for a more conflicting alias, if possible.


Example run command: 
python3.6 -m contextual_embeddings.bootleg_data_prep.add_labels_single_func --processes 20 --data_dir data/aida
'''


import argparse
import glob
import multiprocessing
import os
import random
import shutil
import time
from collections import defaultdict

import jsonlines
import marisa_trie
import ujson as json
from tqdm import tqdm

import bootleg_data_prep.utils.data_prep_utils as prep_utils
from bootleg_data_prep.language import ENSURE_ASCII
from bootleg_data_prep.utils.classes.entity_symbols_prep import EntitySymbolsPrep
from bootleg_data_prep.utils import utils
from bootleg_data_prep.utils.acronym_utils import find_acronyms_from_parentheses, find_manufactured_acronyms, augment_first_sentence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/wiki_dump', help='Directory for data to be saved.')
    parser.add_argument('--filtered_alias_subdir', type=str, default='alias_filtered_sentences', help = 'Subdirectory to save filtered sentences.')
    parser.add_argument('--out_subdir', type = str, default = 'orig_wl', help = 'Where to write processed data to.')
    parser.add_argument('--no_coref', action='store_true', help = 'Turn on to not do coref.')
    parser.add_argument('--no_permute_alias', action='store_true', help = 'Turn on to not use alias swapping. Swapping avoids aliases with a single qid (trivial), and ensures that labeled aliases have the qid in the top 30 (to avoid being dropped).')
    parser.add_argument('--no_acronyms', action='store_true', help = 'Turn on to not do any acronym mining.')
    parser.add_argument('--no_weak_labeling', action='store_true', help = 'If set, will not do ANY weak labeling at all. However, it CAN still do permuting for the gold labels unless you also turn on --no_permute_alias')
    parser.add_argument('--processes', type=int, default=int(0.8*multiprocessing.cpu_count()))
    parser.add_argument('--test', action = 'store_true', help = 'If set, will only generate for one file.')

    args = parser.parse_args()
    return args

def init_process(args):
    qid2singlealias_f,qid2alias_f,aliasconflict_f = args
    global qid2singlealias_global
    global qid2alias_global
    global aliasconflict_global
    qid2singlealias_global = utils.load_json_file(qid2singlealias_f)
    qid2alias_global = utils.load_json_file(qid2alias_f)
    aliasconflict_global = marisa_trie.RecordTrie(fmt="<l").mmap(aliasconflict_f)
    # for qid in qid2alias_global:
    #     qid2alias_global[qid] = set(qid2alias_global[qid])


def launch_subprocess(args, outdir, temp_outdir, qid2singlealias, qid2alias, aliasconflict_f, in_files):
    qid2alias_f = os.path.join(temp_outdir, "qid2alias.json")
    utils.dump_json_file(qid2alias_f, qid2alias)
    qid2singlealias_f = os.path.join(temp_outdir, "qid2singlealias.json")
    utils.dump_json_file(qid2singlealias_f, qid2singlealias)
    all_process_args = [tuple([i+1,
                               len(in_files),
                               outdir,
                               temp_outdir,
                               args,
                               in_files[i],
                               ]) for i in range(len(in_files))]
    pool = multiprocessing.Pool(processes=args.processes, initializer=init_process, initargs=(tuple([qid2singlealias_f,qid2alias_f,aliasconflict_f]),))
    print("Starting pool...")
    for _ in tqdm(pool.imap(subprocess, all_process_args, chunksize=1), total=len(all_process_args)):
        pass
    pool.close()
    return

def choose_new_alias(alias, qid, qid2alias, qid2alias_top30, aliasconflict, doc_ent, sentence_idx):
    # Set a seed to ensure that across ablations, the aliases chosen will be consistent. For example, if we are processing
    # the document for "Q123" and are on sentence 55 of that article, and are currently labeling the QID "Q88" then we will
    # set the seed to 1235588.
    seed = int(str(doc_ent[1:]) + str(sentence_idx) + str(qid[1:]))
    random.seed(seed)
    if qid not in qid2alias:
        return alias
    # If qid is in the top 30 for the alias, and there are at least 2 candidates for that alias, just use that alias
    if qid2alias[qid][alias] and aliasconflict[alias][0][0] > 1:
        return alias
    # Otherwise, find all other aliases for that qid that are in the top 30 and have at least 2 candidates, and randomly choose one
    possible_aliases = [al for al in qid2alias_top30[qid] if aliasconflict[al][0][0] > 1]
    if len(possible_aliases) > 0:
        return random.choice(possible_aliases)
    # We might be in the situation where there are a bunch of aliases for that qid (and the top 30 condition is met) but they
    # all have only 1 candidate. That's better than nothing, so in that case, randomly return one of those aliases.
    if len(qid2alias_top30[qid]) > 0:
        return random.choice(qid2alias_top30[qid])
    # If all of the above fail, then just return the original alias
    return alias


def subprocess(all_args):
    prep_utils.print_memory()
    start_time = time.time()
    idx, total, outdir, temp_outdir, args, in_filepath = all_args
    random.seed(1234)
    # create output files:
    out_fname = os.path.join(outdir, prep_utils.get_outfname(in_filepath))
    out_file = open(out_fname, 'w', encoding='utf8')
    print(f"Starting {idx}/{total}. Reading in {in_filepath}. Outputting to {out_fname}")
    filtered_qid_counts = defaultdict(int)
    filtered_aliases_to_qid_count = defaultdict(lambda: defaultdict(int))
    filtered_aliases_cooccurence = defaultdict(lambda: defaultdict(int))
    filtered_qids_cooccurence = defaultdict(lambda: defaultdict(int))
    no_qid = []
    added_alias_count = 0          # number of weak labels added (not including ones from acronym mining)
    acronym_added_alias_count = 0  # number of weak labels added due to acronym mining
    with jsonlines.open(in_filepath, 'r') as in_file:
        # Make another dict that stores qid -> aliases but only keeps aliases where that qid is in the top 30 results for that alias
        # NOTE THAT this is being created from qid2alias_global, NOT from qid_to_aliases_in_doc. This is because we need qid_to_aliases_in_doc
        # during the weak-label-identifying process, but when swapping out aliases at the end, we can swap to ANY alias for a qid as 
        # long as it's in the top 30. But keep this in mind if you end up using qid_to_aliases_top_30 for a different purpose!
        qid_to_aliases_top_30 = {}
        for q in qid2alias_global:
            qid_to_aliases_top_30[q] = [x for x in qid2alias_global[q] if qid2alias_global[q][x] is True]
        for doc_idx, doc in enumerate(in_file):
            title = doc['title']
            doc_entity = str(doc['qid'])
            if not doc_entity == "-1" and doc_entity in qid2alias_global:
                # print(f"Starting {title} ({doc_entity})")
                aliases_to_qids_in_doc, qid_to_aliases_in_doc = collect_aliases_to_qids_in_doc(doc, qid2alias_global)
                if args.no_coref:
                    aliases_for_finding = marisa_trie.Trie(qid_to_aliases_in_doc[doc_entity])
                    # Allows us to remove aliases if they are also associated frequently with both the doc QID and some other QID in the document
                    # (e.g., the last name of family members ... if the sentence is talking about the parents, children, etc, we don't want them all
                    # labelled as the person in the doc)
                    # LAUREL: do we need to do this anymore?
                    aliases_for_filtering = marisa_trie.Trie(set.difference(set(aliases_to_qids_in_doc.keys()), set(qid_to_aliases_in_doc[doc_entity])))
                else:
                    aliases_for_finding = marisa_trie.Trie(aliases_to_qids_in_doc.keys())
                    # The conflicting aliases have already been pruned away
                    aliases_for_filtering = {}
                max_alias_len = 0
                if len(aliases_to_qids_in_doc) > 0:
                    max_alias_len = max([len(alias.split(" ")) for alias in aliases_to_qids_in_doc.keys()])
                # print(f"Aliases dict {aliases_to_qids_in_doc} \n\nQID Dict {qid_to_aliases_in_doc}")
                new_sentences = []
                possible_acronyms = {} # dictionary of acronym -> tuple(qid, original alias)
                for sentence_idx, sentence in enumerate(doc['sentences']):
                    sentence_str = sentence['sentence']
                    if len(sentence['spans']) > 0 and type(sentence['spans'][0]) is str:
                        sentence['spans'] = [list(map(int, s.split(":"))) for s in sentence["spans"]]
                    # Prep for adding aliases
                    sentence_str = sentence_str.split(" ")

                    if args.no_weak_labeling:
                        found_spans, unswapped_aliases, found_aliases, found_qids, found_sources = [], [], [], [], []
                    else:
                        if not args.no_acronyms:
                            # Check if first sentence of article contains same words as the title of the page; if so, add this so we can use it for acronym mining
                            if sentence_idx == 0 and doc_entity not in sentence['qids']: # don't bother if doc_entity is already linked in the sentence
                                sent_alias_augmented, sent_span_augmented, sent_qid_augmented = augment_first_sentence(sentence['aliases'], sentence['spans'], sentence['qids'], title,
                                                                                                                    sentence_str, args, qid2singlealias_global, qid_to_aliases_in_doc, doc_entity)
                            else: # just copy the originals over
                                sent_alias_augmented, sent_span_augmented, sent_qid_augmented = sentence['aliases'].copy(), sentence['spans'].copy(), sentence['qids'].copy()

                            # Search for acronyms defined in the text in parentheses, e.g. "National Football League (NFL)"
                            paren_acronyms = find_acronyms_from_parentheses(sentence_str, sent_span_augmented, sent_qid_augmented, sent_alias_augmented)
                            for paren_acronym in paren_acronyms:
                                # If acronym is already in possible_acronyms for a DIFFERENT qid, then delete that acronym b/c it's unclear which one is correct, e.g. if
                                # we have both "Australian Broadcasting Company" and "American Bird Conservancy," don't weak-label "ABC" mentions because they're ambiguous.
                                # NOTE: below, [0] indexes into the tuple (qid, alias) returned by find_acronyms_from_parentheses (and by find_manufactured_acronyms)
                                if paren_acronym in possible_acronyms and possible_acronyms[paren_acronym][0] != paren_acronyms[paren_acronym][0]:
                                    del possible_acronyms[paren_acronym]
                                else:
                                    possible_acronyms[paren_acronym] = paren_acronyms[paren_acronym]
                            
                            # Find potential acronyms for each named entity by taking the first letter of each word (e.g. "National Football League" --> "NFL")
                            manufactured_acronyms = find_manufactured_acronyms(sentence_str, sent_span_augmented, sent_qid_augmented, sent_alias_augmented)
                            for manufac_acronym in manufactured_acronyms:
                                # Same as above, if acronym is already in dictionary for a DIFFERENT qid then delete entirely
                                if manufac_acronym in possible_acronyms and possible_acronyms[manufac_acronym][0] != manufactured_acronyms[manufac_acronym][0]:
                                    del possible_acronyms[manufac_acronym]
                                else:
                                    possible_acronyms[manufac_acronym] = manufactured_acronyms[manufac_acronym]

                        # replace spans with a random token -- we don't want aliases to overlap and 
                        # we don't want to 'refind' aliases that we've already identified.
                        error = False
                        for span in sentence['spans']:
                            start, end = span
                            if end > len(sentence_str):
                                error = True 
                                print(span, sentence_str)
                                break
                            for i in range(start, end):
                                sentence_str[i] = '|||'
                        if error:
                            continue
                        sentence_str = ' '.join(sentence_str)
                        # st = time.time()
                        # Find aliases
                        found_aliases, found_spans = prep_utils.find_aliases_in_sentence_tag(sentence_str, aliases_for_finding, max_alias_len)
                        # print("SENTENCe", sentence_str)
                        # print("FOUND", found_aliases)
                        # Find aliases that should be filtered
                        filter_aliases, filter_spans = prep_utils.find_aliases_in_sentence_tag(sentence_str, aliases_for_filtering, max_alias_len)
                        # print("FILTER", filter_aliases)
                        # print(f"Found aliases in {time.time() - st} seconds.")
                        # Filter aliases
                        found_aliases, found_spans = prep_utils.filter_superset_aliases(found_aliases, found_spans, filter_aliases, filter_spans)
                        # print("FINAL", found_aliases)
                        unswapped_aliases = found_aliases[:]
                        found_qids = []
                        for j in range(len(found_aliases)):
                            alias = found_aliases[j]
                            associated_qid = aliases_to_qids_in_doc[alias]
                            alias_conflict = 0
                            if alias in aliasconflict_global:
                                # The tri stores a list of tuples (in our case, each list is len 1 and each tuple has a single int in it)
                                assert len(aliasconflict_global[alias]) == 1
                                assert len(aliasconflict_global[alias][0]) == 1
                                alias_conflict = aliasconflict_global[alias][0][0]
                            # print(f"Alias conflict for {alias} is {alias_conflict}")
                            if args.no_permute_alias:
                                # If not permuting alias, just use the alias. HOWEVER, note that if the qid is not in the top-30 for this alias,
                                # then this label will be DROPPED from the training set later on. So it is likely recommended to leave permuting
                                # alias ON to prevent this loss of information
                                new_alias = alias
                            else:
                                new_alias = choose_new_alias(alias, associated_qid, qid2alias_global, qid_to_aliases_top_30, aliasconflict_global, doc_entity, sentence['doc_sent_idx'])
                                # new_alias = qid2singlealias_global.get(associated_qid, alias)
                                # print(f"Swapping {alias} for {associated_qid} for new alias {new_alias}")
                            added_alias_count += 1
                            found_aliases[j] = new_alias
                            found_qids.append(associated_qid)
                        
                        # Add label for the source/type of the weak label
                        found_sources = []
                        for found_qid in found_qids:
                            if found_qid == doc_entity:
                                found_sources.append('doc_ent') # 'doc_ent' means this label refers to the entity whose Wikipedia page we are on (i.e. the doc entity)
                            else:
                                found_sources.append('coref')                 # added by coref

                        if not args.no_acronyms:
                            # Now for each sentence, check if any of the acronyms in possible_acronyms match any of the words in the sentence 
                            # (case-sensitive, so "WHO" will be assigned to "World Health Organization" but "who" will not).
                            # NOTE: currently, this weak labeling method ONLY works for datasets that have upper/lower case words, and won't work 
                            # if everything is lower case. In that case, a potential solution could be to ensure that acronyms in potential_acronyms 
                            # are NOT regular English words by comparing them to a standard English corpus such as from nltk.corpus.words
                            # First, as we did above for sentence['spans'], replace found spans in the sentence with "|||" so we can ignore them, 
                            # e.g. we don't want to label "NFL" if it is already part of the entity called "2009 NFL Draft"
                            sentence_str = sentence_str.split()
                            for span in found_spans:
                                start, end = span
                                for i in range(start, end):
                                    sentence_str[i] = '|||'
                            # Now check for acronyms in the remaining part of the sentence
                            for word_idx, word in enumerate(sentence_str):                        
                                if word == '|||':
                                    continue
                                # Check if the current word is in possible_acronyms (case sensitive)
                                if word in possible_acronyms:
                                    unswapped_aliases.append(word.lower())
                                    if not args.no_permute_alias: # use alias swapping
                                        if word.lower() in qid_to_aliases_in_doc.get(possible_acronyms[word][0], []):
                                            found_aliases.append(choose_new_alias(word.lower(), possible_acronyms[word][0], qid2alias_global, qid_to_aliases_top_30, aliasconflict_global, doc_entity, sentence['doc_sent_idx']))
                                        else:
                                            found_aliases.append(choose_new_alias(possible_acronyms[word][1], possible_acronyms[word][0], qid2alias_global, qid_to_aliases_top_30, aliasconflict_global, doc_entity, sentence['doc_sent_idx']))
                                        # found_aliases.append(qid2singlealias_global.get(possible_acronyms[word][0]))
                                    else:
                                        # NOTE: that in this else statement, we will NOT permute the alias, so if the qid is not in the top 30 for the alias, 
                                        # then this label will be dropped later on. You should turn ON permute alias if you don't want it to be dropped.
                                        
                                        # Use acronym itself if it already appears in the alias table for the CORRECT qid (but it's not necessarily in the top 30)
                                        if word.lower() in qid_to_aliases_in_doc.get(possible_acronyms[word][0], []):
                                            found_aliases.append(word.lower())
                                        else: # otherwise, use the original alias (again, it's not necessarily in the top 30)
                                            found_aliases.append(possible_acronyms[word][1])
                                    found_spans.append([word_idx, word_idx+1])
                                    found_qids.append(possible_acronyms[word][0])
                                    found_sources.append('acr')
                                    acronym_added_alias_count += 1

                        # Now sort found_aliases, found_qids, found_sources, and found_spans based on found_spans (so they are in order 
                        # of appearance in the sentence)
                        if len(found_spans) > 1:
                            zipped = list(zip(found_spans, unswapped_aliases, found_aliases, found_qids, found_sources))
                            zipped.sort(key = lambda x: x[0][0])
                            found_spans, unswapped_aliases, found_aliases, found_qids, found_sources = [list(x) for x in zip(*zipped)]

                    # We merge the new sentence together
                    #TODO: We can just sort based on the index of the span. this is much faster.
                    new_aliases = []
                    new_unswapped_aliases = []
                    new_spans = [] 
                    new_qids = []
                    sources = []
                    from_gold = []
                    old_index = 0 
                    found_index = 0 
                    sent_len = len(sentence['sentence'].split(" "))
                    while old_index + found_index < len(found_aliases) + len(sentence['aliases']):
                        if old_index < len(sentence['aliases']):
                            old_span_start = sentence['spans'][old_index][0]
                        else:
                            old_span_start = sent_len
                        if found_index < len(found_aliases):
                            found_span_start = found_spans[found_index][0]
                        else:
                            found_span_start = sent_len
                        if old_span_start < found_span_start:
                            # Permute aliases for the gold labels (previously we only permuted the found labels)
                            temp_alias = sentence['aliases'][old_index]
                            new_unswapped_aliases.append(temp_alias)
                            if args.no_permute_alias:
                                new_aliases.append(temp_alias)
                            else:
                                new_aliases.append(choose_new_alias(temp_alias, sentence['qids'][old_index], qid2alias_global, qid_to_aliases_top_30, aliasconflict_global, doc_entity, sentence['doc_sent_idx']))
                            
                            new_qids.append(sentence['qids'][old_index])
                            new_spans.append(sentence['spans'][old_index])
                            sources.append('gold')
                            old_index += 1
                            from_gold.append(True)
                        else:
                            new_unswapped_aliases.append(unswapped_aliases[found_index])
                            # Don't need to worry about permuting aliases here, because that has already been done above for all found aliases
                            new_aliases.append(found_aliases[found_index])
                            new_qids.append(found_qids[found_index])
                            new_spans.append(found_spans[found_index])
                            sources.append(found_sources[found_index])
                            found_index += 1
                            from_gold.append(False)
                    new_sentences.append({
                        'doc_sent_idx': sentence['doc_sent_idx'],
                        'sentence': sentence['sentence'],
                        'aliases': new_aliases,
                        'unswap_aliases': new_unswapped_aliases,
                        'spans': new_spans,
                        'qids': new_qids,
                        'gold': from_gold,
                        'sources': sources
                    })
                doc['sentences'] = new_sentences
            else: 
                no_qid.append(title)
                for i in range(len(doc['sentences'])):
                    doc['sentences'][i]['gold'] = [True for _ in range(len(doc['sentences'][i]['aliases']))]
                    doc['sentences'][i]['unswap_aliases'] = doc['sentences'][i]['aliases'][:]
            # update stats
            for i in range(len(doc["sentences"])):
                aliases = doc["sentences"][i]["aliases"]
                qids = doc["sentences"][i]["qids"]
                for qid in qids:
                    filtered_qid_counts[qid] += 1
                if len(aliases) > 0:
                    for alias, qid in zip(aliases, qids):
                        filtered_aliases_to_qid_count[alias][qid] += 1
                    for j, alias in enumerate(aliases):
                        for k in range(j+1, len(aliases)):
                            filtered_aliases_cooccurence[alias][aliases[k]] += 1
                            filtered_aliases_cooccurence[aliases[k]][alias] += 1
                    for j, qid in enumerate(qids):
                        for k in range(j+1, len(qids)):
                            filtered_qids_cooccurence[qid][qids[k]] += 1
                            filtered_qids_cooccurence[qids[k]][qid] += 1
            out_file.write(json.dumps(doc, ensure_ascii=ENSURE_ASCII) + '\n')
    out_file.close()
    print(f"LEN: {len(filtered_qid_counts)} LEN: {len(filtered_aliases_to_qid_count)} LEN: {len(filtered_aliases_cooccurence)}")
    utils.dump_json_file(os.path.join(temp_outdir, f"filtered_alias_to_qid_count_{idx}.json"), filtered_aliases_to_qid_count)
    utils.dump_json_file(os.path.join(temp_outdir, f"filtered_qid_counts_{idx}.json"), filtered_qid_counts)
    utils.dump_json_file(os.path.join(temp_outdir, f"filtered_aliases_cooccurence_{idx}.json"), filtered_aliases_cooccurence)
    utils.dump_json_file(os.path.join(temp_outdir, f"filtered_qids_cooccurence_{idx}.json"), filtered_qids_cooccurence)
    print(f"Finished {idx}/{total}. Written to {out_fname}. {time.time() - start_time} seconds. Added a total of {added_alias_count} aliases \
    (not including acronym mining). Added a total of {acronym_added_alias_count} aliases through acronym mining.")
    return None

def collect_aliases_to_qids_in_doc(doc, qid2alias_global):
    """
    :param doc:
    :param qid2alias_global:
    :return: aliases_to_qids_in_doc_pruned, qid_to_aliases_in_doc_pruned

    The method gathers a dict of alias->qid->count for all qids linked to in a given document, including the aliases that refer to the QID of the Wikipedia page itself.
    These are then pruned to remove aliases that have different qids that appear with similar frequencies (a sign of a noisy alias).
    """
    aliases_to_qids_in_doc = defaultdict(lambda: defaultdict())
    title = doc['title']
    doc_entity = str(doc['qid'])
    assert doc_entity != "-1"
    assert doc_entity in qid2alias_global
    # Add aliases pointing to the document
    aliases = qid2alias_global[doc_entity].keys()
    for al in aliases:
        assert len(al) > 0
        top30_bool = qid2alias_global[doc_entity][al] # we want this boolean value to pass through without being dropped
        # We correct this count below when pruning
        aliases_to_qids_in_doc[al][doc_entity] = [1, top30_bool]
    # We always add other aliases so when we prune, we can remove highly conflicting aliases to use during weak labelling
    for sentence in doc['sentences']:
        sentence_qids = sentence['qids']
        # Update the aliases_to_qids_in_doc
        for qid in sentence_qids:
            for al in qid2alias_global.get(qid, set()):
                top30_bool = qid2alias_global[qid][al] # again just let this pass through without being dropped
                if qid not in aliases_to_qids_in_doc[al]:
                    aliases_to_qids_in_doc[al][qid] = [0, top30_bool]
                aliases_to_qids_in_doc[al][qid][0] += 1
    return prune_aliases_to_qids_in_doc(doc_entity, aliases_to_qids_in_doc)

def prune_aliases_to_qids_in_doc(doc_entity, aliases_to_qids_in_doc):
    """doc_entity: QID of page we are on
       aliases_to_qids_in_doc: list of aliases on page -> QID they link to UNION all aliases that point to doc_entity -> doc_entity"""
    aliases_to_qids_in_doc_pruned = {}
    qid_to_aliases_in_doc_pruned = defaultdict(lambda: defaultdict())
    total_qid_count = sum(v[0] for qid_dict in aliases_to_qids_in_doc.values() for v in qid_dict.values())
    # print(f"Total Count for {doc_entity} is {total_qid_count}")
    # We want to assign some weight of doc_entity aliases -> doc_entity (they are often not actual link in Wikipedia so get a low weight by default)
    doc_entity_perc_of_total = 0.2
    # This is representing the popularity of the doc entity in a document. If there is some other QID that appears with some alias
    # also associated with the doc_entity more than doc_entity_count times, we remove this highly conflicting alias from consideration
    popularity_threshold = 5
    doc_entity_count = max(doc_entity_perc_of_total*total_qid_count, 1.0)
    for al in aliases_to_qids_in_doc:
        qid_dict = aliases_to_qids_in_doc[al]
        # qid_dict is qid -> list([count of number of times al linked to qid, top_30_boolean]); if only one qid, add it for that alias
        # otherwise, find the most popular qid if one exists
        if len(qid_dict) == 1:
            qid_to_add = next(iter(qid_dict.keys()))
            top30_bool = qid_dict[qid_to_add][1]
            # Add the qid to the list of aliases
            aliases_to_qids_in_doc_pruned[al] = qid_to_add
            qid_to_aliases_in_doc_pruned[qid_to_add][al] = top30_bool
        else:
            # Assign the doc entity weight
            if doc_entity in qid_dict:
                qid_dict[doc_entity][0] = doc_entity_count
            sorted_qids = sorted(qid_dict.items(), key=lambda x: x[1][0], reverse=True)
            # Only add if count is above threshold
            if sorted_qids[0][1][0] > popularity_threshold*sorted_qids[1][1][0]:
                qid_to_add = sorted_qids[0][0]
                top30_bool = sorted_qids[0][1][1]
                aliases_to_qids_in_doc_pruned[al] = qid_to_add
                qid_to_aliases_in_doc_pruned[qid_to_add][al] = top30_bool
    return aliases_to_qids_in_doc_pruned, qid_to_aliases_in_doc_pruned


def get_qid2aliases(alias2qids, entity_dump, out_dir):
    """
    :param entity_dump:
    :param out_dir:
    :return: qid2singlealias a mapping a qid to a single alias; alias is chosen to be the one that has the largest candidate list (most conflicting)
    :return: aliasconflict_f is a memmapped trie (aka a dictionary) from alias to len of that aliases candidate list (i.e., the amount of conflict for that alias)
    :return: qid2alias a mapping of qid -> aliases -> boolean value. Basically, each qid is mapped to all of its aliases, and each of those aliases
             is then mapped to a boolean value. The boolean value indicates whether or not the qid is in the top 30 candidates for that alias. This is important
             to know, because if the qid is not in the top 30 for that alias, then it will essentially be dropped during weak labeling and training. For example,
             there are many people with the alias "smith." For someone like Ozzie Smith, who is NOT in the top-30 Smiths, we still want to weak label mentions of
             "Smith" on his page, but we don't want to actually use "smith" as the alias because it will NEVER return Ozzie Smith during candidate generation.
    """
    # EntitySymbol stores a map of alias-QIDs. We need to build a map of QID->alias
    MAX_CANDS = 30
    qid2alias = {}
    aliasconflict_keys = []
    aliasconfict_values = []
    aliasconflict_f = os.path.join(out_dir, f'marisa_conflict.marisa')
    temp_qid2alias_single = {}
    for alias in alias2qids.keys():
        if alias not in entity_dump.get_all_aliases():
            continue
        assert len(alias) > 0
        cands = [q[0] for q in alias2qids[alias] if q[0] in entity_dump.get_qid_cands(alias)]
        if len(cands) == 0:
            continue
        aliasconflict_keys.append(alias)
        # Tries need iterable values
        aliasconfict_values.append(tuple([len(cands)]))
        for idx, qid in enumerate(cands):
            if not qid in qid2alias:
                qid2alias[qid] = {}
            if idx < MAX_CANDS:
                qid2alias[qid][alias] = True # True means this qid is in the top 30 candidates for this alias
            else:
                qid2alias[qid][alias] = False # False means this qids is NOT in the top 30 candidates for this alias,
                                              # so later we will need to swap it for a different alias so it doesn't get dropped
            # Only use an alias in qid2singlealias if the qid is in the top 30 results for that alias, otherwise it's pointless b/c it will be dropped
            if idx < MAX_CANDS:
                if not qid in temp_qid2alias_single:
                    temp_qid2alias_single[qid] = []
                temp_qid2alias_single[qid].append([alias, len(cands)])
            
    qid2singlealias = {}
    for qid in temp_qid2alias_single:
        aliases = sorted(temp_qid2alias_single[qid], key = lambda x: x[1], reverse=True)
        # Occasionally we get highly conflicting all numeric aliases for a qid (like years).
        # We don't want to replace years for QIDs
        non_num_aliases = list(filter(lambda x: not x[0].isnumeric(), aliases))
        if len(non_num_aliases) > 0:
            new_alias = non_num_aliases[0][0]
        else:
            new_alias = aliases[0][0]
        qid2singlealias[qid] = new_alias
    fmt = "<l"
    trie = marisa_trie.RecordTrie(fmt, zip(aliasconflict_keys, aliasconfict_values))
    trie.save(aliasconflict_f)
    return qid2singlealias, aliasconflict_f, qid2alias


# Here, we just copy the entity dump over to the new directory but delete erroneous -1 qids
# that crop up if we don't have the page title in our mapping.
def modify_counts_and_dump(args, filtered_aliases_to_qid, entity_dump):
    alias2qids = entity_dump.get_alias2qids_dict()
    qid2title = entity_dump.get_qid2title_dict()
    if "-1" in qid2title:
        del qid2title["-1"]

    max_candidates = entity_dump.max_candidates

    for al in alias2qids:
        all_pairs = alias2qids[al]
        qids = [p[0] for p in all_pairs]
        if "-1" in qids:
            print(f"BAD: for alias {al} there is a -1 QID of {alias2qids[al]}. Will remove")
            all_pairs.pop(qids.index("-1"))
            alias2qids[al] = all_pairs

    # Make entity dump object
    entity_dump = EntitySymbolsPrep(
        max_candidates=max_candidates,
        alias2qids=alias2qids,
        qid2title=qid2title
    )
    out_dir = os.path.join(args.data_dir, args.out_subdir, 'entity_db/entity_mappings')
    entity_dump.save(out_dir)

def main():
    gl_start = time.time()
    multiprocessing.set_start_method("forkserver", force=True)
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    outdir = prep_utils.get_outdir(args.data_dir, args.out_subdir, remove_old=True)
    temp_outdir = prep_utils.get_outdir(os.path.join(args.data_dir, args.out_subdir), "_temp", remove_old=True)

    # get inputs files 
    path = os.path.join(args.data_dir, args.filtered_alias_subdir, "*.jsonl")
    in_files = prep_utils.glob_files(path)
    # if in test mode, just take a single input file
    if args.test:
        in_files = in_files[:1]

    # this loads all entity information (aliases, titles, etc)
    entity_dump = EntitySymbolsPrep.load_from_cache(load_dir=os.path.join(args.data_dir, args.filtered_alias_subdir, 'entity_db/entity_mappings'))
    print(f"Loaded entity dump with {entity_dump.num_entities} entities.")

    # Load wikidata alias2qid
    alias2qid = entity_dump.get_alias2qids_dict()
    # generates mappings from qids to aliases and a memmaped trie (aka dict) from alias to length of that aliases candidate list (measure of how conflicting that alias is)
    qid2singlealias, aliasconflict_f, qid2alias = get_qid2aliases(alias2qid, entity_dump, temp_outdir)
    print(f"Created qid2alias map over {len(qid2alias)} QIDs.")
    # launch subprocesses and collect outputs
    print(f"Loaded {len(in_files)} files from {path}. Launching {args.processes} processes.")
    launch_subprocess(args, outdir, temp_outdir, qid2singlealias, qid2alias, aliasconflict_f, in_files)

    # Gather new counts
    # Total QID count
    qid_count_files = glob.glob(f"{temp_outdir}/filtered_qid_counts_*")
    list_of_qids_dicts = [utils.load_json_file(f) for f in qid_count_files]
    filtered_qid_count = prep_utils.aggregate_list_of_dictionaries(list_of_qids_dicts)
    # Alias, qid pair counts
    aliases_to_qid_count_files = glob.glob(f"{temp_outdir}/filtered_alias_to_qid_count_*")
    list_of_alias_dicts = [utils.load_json_file(f) for f in aliases_to_qid_count_files]
    filtered_aliases_to_qid = prep_utils.aggregate_list_of_nested_dictionaries(list_of_alias_dicts)
    # Alias co-occurrence count
    aliases_to_alias_count_files = glob.glob(f"{temp_outdir}/filtered_aliases_cooccurence_*")
    list_of_alias_dicts = [utils.load_json_file(f) for f in aliases_to_alias_count_files]
    filtered_aliases_to_alias = prep_utils.aggregate_list_of_nested_dictionaries(list_of_alias_dicts)
    # QID co-occurrence count
    qids_to_qid_count_files = glob.glob(f"{temp_outdir}/filtered_qids_cooccurence_*")
    list_of_qids_dicts = [utils.load_json_file(f) for f in qids_to_qid_count_files]
    filtered_qids_to_qids = prep_utils.aggregate_list_of_nested_dictionaries(list_of_qids_dicts)

    # Save counts
    utils.dump_json_file(os.path.join(outdir, "filtered_qid_count.json"), filtered_qid_count)
    utils.dump_json_file(os.path.join(outdir, "filtered_aliases_to_qid_count.json"), filtered_aliases_to_qid)
    utils.dump_json_file(os.path.join(outdir, "filtered_aliases_to_alias.json"), filtered_aliases_to_alias)
    utils.dump_json_file(os.path.join(outdir, "filtered_qids_to_qids.json"), filtered_qids_to_qids)
    modify_counts_and_dump(args, filtered_aliases_to_qid, entity_dump)
    # remove temp
    shutil.rmtree(temp_outdir)
    vars(args)["out_dir"] = outdir
    prep_utils.save_config(args, "add_labels_single_func_config.json")
    print(f"Finished add_labels_single_func in {time.time() - gl_start} seconds.")





if __name__ == '__main__':
    main()
