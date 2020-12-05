
LAUREL::::SEND THE PID FILE, TYPES IN OTHER LANGUAGES WIKIDATA, DISMBIGUATION FILTER, WIKIDATA QIDS MISSING...MIGHT NEED ENGLISH

0. Download wikidata and process.
`python3 -m processor.process_dump --input_file /lfs/raiders8/0/lorr1/wikidata/raw_data/latest-all.json --out_dir /lfs/raiders8/0/lorr1/wikidata_sv/processed_batches --language_id sv --num_processes 40`


1. Run the wiki extractor code. This will gen

```
ls
>>> svwiki-20201120-pages-articles-multistream.xml
```
`python3 -m bootleg_data_prep.wiki_extractor --output sentences --output2 pageids svwiki-20201120-pages-articles-multistream.xml --processes 5 &> wiki_extractor.out`

This will output
```
INFO: Finished 5-process extraction of 3552747 articles in 14728.0s (241.2 art/s)
INFO: total of page: 3988933, total of articl page: 3552747; total of used articl page: 3552747
Final end time 14728.023938179016s
```

We expect `ls sv_data` to output
`pageids  sentences  svwiki-20201120-pages-articles-multistream.xml`

2. Get mapping of all wikipedia ids to QIDs
`python3 -m processor.bootleg.get_title_to_ids \
    --data /lfs/raiders8/0/lorr1/wikidata_sv \
    --wikipedia_xml /lfs/raiders8/0/lorr1/sv_data/svwiki-20201120-pages-articles-multistream.xml \
    --total_wikipedia_xml_lines 436717867 \
    --wikipedia_pageids /lfs/raiders8/0/lorr1/sv_data/pageids/AA \
    --out_dir /lfs/raiders8/0/lorr1/sv_data/title_mappings`

This took about 1240 sec to run. You should now have `title_mappings` in `sv_data`.

(Optional) If you want to add wikidata aliases. Run
`python3 -m processor.bootleg.create_aliases \
    --qids '' \
    --data /lfs/raiders8/0/lorr1/wikidata_sv \
    --out_file /lfs/raiders8/0/lorr1/sv_data/augmented_alias_map_large.jsonl`
    
If you run this, you should now have `augmented_alias_map_large.jsonl` in `sv_data`.

3. Curate aliases
- `python3 -m bootleg_data_prep.curate_aliases \
    --min_frequency 2 \
    --sentence_dir /lfs/raiders8/0/lorr1/sv_data/sentences \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --wd_aliases /lfs/raiders8/0/lorr1/sv_data/augmented_alias_map_large.jsonl \
    --title_to_qid /lfs/raiders8/0/lorr1/sv_data/title_mappings/title_to_all_ids.jsonl \
    --processes 10`

- `python3 -m bootleg_data_prep.remove_bad_aliases \
    --sentence_dir /lfs/raiders8/0/lorr1/sv_data/sentences \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --title_to_qid /lfs/raiders8/0/lorr1/sv_data/title_mappings/title_to_all_ids.jsonl \
    --benchmark_qids '' \
    --processes 10`

Output expected is `alias_filtered_sentences  curate_aliases  curate_aliases_config.json  remove_bad_aliases_config.json` inside `sv_data`.

4. Extract KG and Types
`python3 -m processor.bootleg.get_all_wikipedia_triples \
    --data /lfs/raiders8/0/lorr1/wikidata_sv \
    --out_dir wikidata_output \
    --processes 10 \
    --qids /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json`
`python3 -m processor.bootleg.create_kg_adj \
    --data /lfs/raiders8/0/lorr1/wikidata_sv \
    --out_dir wikidata_output \
    --processes 10 \
    --qids /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json`
`python3 -m processor.bootleg.get_types \
    --data /lfs/raiders8/0/lorr1/wikidata_sv \
    --out_dir wikidata_output \
    --processes 10 \
    --qids /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump/alias_filtered_sentences/entity_db/entity_mappings/qid2title.json`

Then run
```
mkdir /lfs/raiders8/0/lorr1/sv_data/embs
cp /lfs/raiders8/0/lorr1/wikidata_sv/wikidata_output/kg_triples.txt /lfs/raiders8/0/lorr1/sv_data/embs/kg_triples_1129.txt
cp /lfs/raiders8/0/lorr1/wikidata_sv/wikidata_output/kg_adj.txt /lfs/raiders8/0/lorr1/sv_data/embs/kg_adj_1129.txt
cp /lfs/raiders8/0/lorr1/wikidata_sv/wikidata_output/wikidatatitle_to_typeqid.json /lfs/raiders8/0/lorr1/sv_data/embs/wikidatatitle_to_typeqid_1129.json
cp /lfs/raiders8/0/lorr1/wikidata_sv/wikidata_output/wikidatatitle_to_typeid.json /lfs/raiders8/0/lorr1/sv_data/embs/wikidatatitle_to_typeid_1129.json
cp /lfs/raiders8/0/lorr1/wikidata_sv/wikidata_output/wikidata_types.json /lfs/raiders8/0/lorr1/sv_data/embs/wikidata_types_1129.json
```
Once things are in place, you can generate our relation type mapping.
- `python3 -m processor.bootleg.get_relation_types_from_triples \
    --pid_file /dfs/scratch0/lorr1/bootleg/data/wikidata_mappings/pid_names.json \
    --kg_triples /lfs/raiders8/0/lorr1/sv_data/embs/kg_triples_1129.txt \
    --type_file /lfs/raiders8/0/lorr1/sv_data/embs/wikidata_types_1129.json \
    --output_file /lfs/raiders8/0/lorr1/sv_data/embs/kg_relation_types_1129.json \
    --output_vocab_file /lfs/raiders8/0/lorr1/sv_data/embs/relation_to_typeid_1129.json`

At the end, you should have
`ls sv_data/embs`
`kg_adj_1129.txt  kg_relation_types_1129.json  kg_triples_1129.txt  relation_to_typeid_1129.json  wikidatatitle_to_typeid_1129.json  wikidatatitle_to_typeqid_1129.json  wikidata_types_1129.json`

Note that if you'd like to filter the type system or something else, you can. We have a notebook in notebooks/filter_types_wikidata.ipynb that does this. We also support adding a coarser grained type system (we use Hyena) if desired. They are left blank by default.

(Optional weak labeling)

5. Filter data
# subfolder_name is whatever you want to call the folder of the data
`python3 -m bootleg_data_prep.data_filter \
    --processes 10 \
    --train_in_candidates \
    --subfolder_name full_wiki \
    --max_candidates 30 \
    --sentence_filter_func sentence_filter_short \
    --orig_dir alias_filtered_sentences \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --disambig_file ''`

6. Split Data
This will split data by Wikipedia page (by default).

`python3 -m bootleg_data_prep.merge_shuff_split \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --subfolder_name full_wiki \
    --split 10`

7. Slice data (optional)
- `python3 -m bootleg_data_prep.prep_generate_slices_part1 \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --subfolder_name full_wiki \
    --processes 10 \
    --multilingual`
- `python3 -m bootleg_data_prep.prep_generate_slices_part2 \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --subfolder_name full_wiki \
    --processes 10 \
    --emb_dir /lfs/raiders8/0/lorr1/sv_data/embs \
    --kg_adj kg_adj_1129.txt \
    --kg_triples kg_triples_1129.txt \
    --hy_vocab ''\
    --hy_types ''\
    --wd_vocab wikidatatitle_to_typeid_1129.json \
    --wd_types wikidata_types_1129.json \
    --rel_vocab relation_to_typeid_1129.json \
    --rel_types kg_relation_types_1129.json \
    --multilingual`
- `python3 -m bootleg_data_prep.generate_slices \
    --subfolder_name full_wiki \
    --data_dir /lfs/raiders8/0/lorr1/sv_data/data/wiki_dump \
    --processes 10 \
    --emb_dir /lfs/raiders8/0/lorr1/sv_data/embs \
    --kg_adj kg_adj_1129.txt \
    --kg_triples kg_triples_1129.txt \
    --hy_vocab ''\
    --hy_types ''\
    --wd_vocab wikidatatitle_to_typeid_1129.json \
    --wd_types wikidata_types_1129.json \
    --rel_vocab relation_to_typeid_1129.json \
    --rel_types kg_relation_types_1129.json`




