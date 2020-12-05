# This is a readme for generating 1) the AIDA dataset and 2) a corresponding dataset from wikipedia for pretraining

Process: 

```
# Extract unfiltered AIDA data from raw TSV -- handles WPID-> QID mapping 
python3.6 -m contextual_embeddings.data_prep.benchmarks.aida.build_aida_datasets

# Build Wikipedia dataset for pretraining in aida directory 
python3.6 -m contextual_embeddings.data_prep.curate_aliases --data_dir data/aida
python3.6 -m contextual_embeddings.data_prep.remove_bad_aliases --data_dir data/aida
python3.6 -m contextual_embeddings.data_prep.add_labels_single_func --processes 20 --data_dir data/aida
python3.6 -m contextual_embeddings.data_prep.data_filter --max_candidates 30 --processes 20 --train_in_candidates --sentence_filter_func sentence_filterQID --subfolder_name aidawiki --qid_filter_file data/aida/aida_qids.json --data_dir data/aida
python3.6 -m contextual_embeddings.data_prep.merge_shuff_split --data_dir data/aida --subfolder_name aidawiki
python3.6 -m contextual_embeddings.data_prep.generate_slices --processes 1 --kg_adj kg_adj.txt --data_dir data/aida --subfolder_name aidawiki
```

For this last step, make sure `kg_adj.txt` is in `<emb_dir>`.