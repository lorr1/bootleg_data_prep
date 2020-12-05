We get <lang>_aliases.jsonl, <lang>_wp_to_wd.jsonl, input/ where input has a set of jsonl files ready for parsing.

First extract a set of all qids for this language.

'''
lang = "de"
out_f = open(f"{lang}_qids.json","w")
in_f = open(f"{lang}_wp_to_wd.jsonl", "r")
import json
all_qids = set()
for line in in_f:
    obj = json.loads(line.strip())
    all_qids.add(obj["wd_id"])



json.dump(list(all_qids), out_f)
'''

Then run the following from wikidata_processor to get all kg relationships and types (alter the arg input to be the qid file generated above for each lanuguage)

- `export lang=de`
- `python3.6 -m processor.get_all_wikipedia_triples --qids /lfs/raiders8/0/lorr1/${lang}_bootleg/${lang}_qids.json`
- `python3.6 -m processor.create_kg_adj --qids /lfs/raiders8/0/lorr1/${lang}_bootleg/${lang}_qids.json`
- `python3.6 -m processor.get_types --qids /lfs/raiders8/0/lorr1/${lang}_bootleg/${lang}_qids.json`

Go back to where the language wikipedia data is
- `mkdir embs_${lang}`
- `cp /lfs/raiders8/0/lorr1/wikidata/kg_output/*.txt embs_${lang}`
- `cp /lfs/raiders8/0/lorr1/wikidata/kg_output/wikidata_to_typeid.json  embs_${lang}`
- `python3 -m contextual_embeddings.data_prep.multilingual.filter_types`
- `python3 -m contextual_embeddings.data_prep.multilingual.gen_wiki_page_adj`
- `python3 -m contextual_embeddings.data_prep.multilingual.gen_sent_cooc`
- `python3 -m contextual_embeddings.data_prep.multilingual.gen_entity_dump`
- `python3 -m contextual_embeddings.data_prep.merge_shuff_split --subfolder_name raw --data_dir ${lang}_bootleg --with_test_dev_augment --split 5 --hash_keys sent_idx_unq`
- `python3 -m contextual_embeddings.data_prep.multilingual.collect_stats`
- `python3 -m contextual_embeddings.data_prep.prep_generate_slices_part1 --subfolder_name raw --data_dir ${lang}_bootleg --processes 32 --multilingual`
- `python3 -m contextual_embeddings.data_prep.prep_generate_slices_part2 --subfolder_name raw --data_dir ${lang}_bootleg --extras_dir ${lang}_bootleg/extas --processes 90 --emb_dir ${lang}_bootleg/embs_${lang} --wd_vocab wikidata_to_typeid_filt.json --kg_adj kg_adj.txt --overwrite --multilingual`
- `python3 -m contextual_embeddings.data_prep.generate_slices --subfolder_name raw --data_dir ${lang}_bootleg --extras_dir ${lang}_bootleg/extas --processes 90 --emb_dir ${lang}_bootleg/embs_${lang} --wd_vocab wikidata_to_typeid_filt.json --kg_adj kg_adj.txt --overwrite` with or without `--dev_filters no_head tail_only`