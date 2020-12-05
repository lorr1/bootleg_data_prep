## AIDA Doc Data Generation with Pershina Candidate List

1. Generate the **Pershina candidate list** by running the [notebook](PershinaCands.ipynb). The original data was copied to the directory `raw` from the [paper Github repo](`https://github.com/masha-p/PPRforNED/tree/master/AIDA_candidates`). The notebook should save the candidate list to the directory `processed`.


2. Generate the **AIDA doc data** by running the following command. This will add the title to each sentence to represent the document. It will also add a [SEP] token between the title and the sentence. Aliases in the title will be added, but with false anchors:
```
python3.6 -m bootleg_data_prep.benchmarks.aida.build_aida_datasets --scope sentence --include_title --include_aliases_in_prefix
```

3. Convert the AIDA doc data to the Pershina format, that is where each alias has a unique identifier. To do this, run
```
python3.6 -m bootleg_data_prep.benchmarks.pershina.convert_dataset_pershina --data_dir data/aida_title_include_aliases_in_prefix/unfiltered/ --out_dir data/aida_docwiki_pershina/unfiltered
```

4. Now the AIDA doc data is ready to be filtered by the Wikipedia entity dump. To do this, run
```
python3.6 -m bootleg_data_prep.benchmarks.filter_and_compute_recall --data data/aida_docwiki_pershina --entity_dump /dfs/scratch1/mleszczy/bootleg/data/wiki_0906/entity_db/entity_mappings/ --method aida --aida_candidates bootleg_data_prep/benchmarks/pershina/processed/cands.json
```

AIDA doc data should be ready for evaluation!