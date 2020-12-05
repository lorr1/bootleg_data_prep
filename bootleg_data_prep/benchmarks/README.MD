# Generating benchmark data

## 1. Creating raw dataset
We need to convert the raw benchmark data to our JSONL format and map entity references to QID identifiers. Each dataset has a specific file to do this. In the future, we might just want to consolidate these. Oh well. 

```
# AIDA
python3.6 -m contextual_embeddings.data_prep.benchmarks.aida.build_aida_datasets 

# Kore50
python3.6 -m contextual_embeddings.data_prep.benchmarks.kore50.build_kore50_datasets

# MSNBC
python3.6 -m contextual_embeddings.data_prep.benchmarks.msnbc.build_msnbc_data

# RSS500
python3.6 -m contextual_embeddings.data_prep.benchmarks.rss500.build_rss500_dataset
```

By default, most of these scripts create a folder under data specific to the benchmark, and write the processed jsonl files under the subfolder unfiltered (i.e. `data/kore50/unfiltered/test.jsonl`)


## 2. Running model. 
Evaluating a model on a benchmark requires the following steps: 

1. Filtering the benchmark data according to the entity_dump -- we only keep mentions where the correct entity is contained in our candidate list for the respective entity. We also compute candidate recall here. 
2. If applicable, finetuning the model on the training data for the benchmark. 
3. Getting predictions over the benchmark. 
4. Running csv.py and getting evaluation outputs. 

Fortunately, we've compressed this all into one script: `run_benchmarks.py`

```
usage: run_benchmarks.py [-h] [--model_to_run MODEL_TO_RUN]
                         [--model_name MODEL_NAME]
                         [--save_directory SAVE_DIRECTORY]
                         [--default_test_only_json DEFAULT_TEST_ONLY_JSON]
                         [--pretrain_data PRETRAIN_DATA]
                         [--benchmark {kore50,msnbc,rss500,aida}]
                         [--use_ganea] [--ganea_candidates GANEA_CANDIDATES]
                         [--overwrite_model] [--overwrite_filtered_data]
                         [--emb_dir EMB_DIR] [--lr LR] [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --model_to_run MODEL_TO_RUN
                        Path to top level file with model. (default: )
  --model_name MODEL_NAME
  --save_directory SAVE_DIRECTORY
                        Directory to save runs to. (default: benchmark_runs)
  --default_test_only_json DEFAULT_TEST_ONLY_JSON
                        Skeleton for benchmarks with only test sets. (default:
                        test_only_skeleton.json)
  --pretrain_data PRETRAIN_DATA
                        Path to data used to pretrain model. Must have entity
                        dump! (default: )
  --benchmark {kore50,msnbc,rss500,aida}
                        Benchmark to use. (default: kore50)
  --use_ganea           whether to use the Ganea candidate list when
                        finetuning (default: False)
  --ganea_candidates GANEA_CANDIDATES
                        Path to ganea candidates file. (default: contextual_em
                        beddings/data_prep/benchmarks/ganea/processed/cands.js
                        on)
  --overwrite_model     Whether to overwrite the saved model. (default: False)
  --overwrite_filtered_data
                        Whether to overwrite the filtered data. (default:
                        False)
  --emb_dir EMB_DIR     Path to embedding directory. (default:
                        /dfs/scratch1/mleszczy/contextual-embeddings-git/embs)
  --lr LR               learning rate to use for finetuning (default: 1e-05)
  --epochs EPOCHS       number of epochs to finetune for. (default: 10)
```

Example usage: 
```
python3.6 -m contextual_embeddings.data_prep.benchmarks.run_benchmarks \
	--pretrain_data /dfs/scratch1/mleszczy/contextual-embeddings-git/data/wiki_final/ \
	--model_to_run /dfs/scratch1/mleszczy/contextual-embeddings-git/kore50_runs/kg_full_mask/20200410_070121/ \
	--model_name model1 \
	--benchmark kore50 
```
The above script will: 
- fetch  `model1.pt` from `/dfs/scratch1/mleszczy/contextual-embeddings-git/kore50_runs/kg_full_mask/20200410_070121/`
- filter the kore50 data according to the entity dump in `/dfs/scratch1/mleszczy/contextual-embeddings-git/data/wiki_final/`. 
- for the AIDA benchmark, this script also finetunes the model on the passed training data and saves it.
- evaluate the model on the filtered data, and save results to `benchmark_runs/kore50/20200410_070121`. 
