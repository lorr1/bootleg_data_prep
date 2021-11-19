source ../../venv/bin/activate
source ./benchmarks_envs.bash

KORE50_BENCHMARK_OUTPUT=$BOOTLEG_BENCHMARK_OUTPUT/kore50

rm -rf $KORE50_BENCHMARK_OUTPUT
mkdir -p $BOOTLEG_BENCHMARK_OUTPUT
mkdir -p $KORE50_BENCHMARK_OUTPUT

cd $BOOTLEG_CODE_DIR

python3 ./bootleg_data_prep/benchmarks/kore50/build_kore50_datasets.py \
    --out_dir $KORE50_BENCHMARK_OUTPUT \
    --sub_dir unfiltered \
    --output_format jsonl \
    --scope sentence \
    --title_to_qid $BOOTLEG_WIKIPEDIA_DIR/title_mappings/title_to_all_ids.jsonl \
    --dataset ./bootleg_data_prep/benchmarks/kore50/raw_data/kore50_$BOOTLEG_LANG_CODE.txt \
    --langid $BOOTLEG_LANG_CODE


python3 -m bootleg_data_prep.data_filter \
    --processes $BOOTLEG_PROCESS_COUNT \
    --train_in_candidates \
    --subfolder_name kore50_aliases_filtered \
    --max_candidates 30 \
    --sentence_filter_func sentence_filterQID \
    --orig_dir alias_filtered_sentences \
    --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
    --filter_file $KORE50_BENCHMARK_OUTPUT/kore50_qids.json \
    --benchmark_qids ''

python3 -m bootleg_data_prep.merge_shuff_split \
    --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
    --subfolder_name kore50_aliases_filtered \
    --split 10

python3 -m bootleg_data_prep.prep_generate_slices_part1 \
  --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
  --subfolder_name kore50_aliases_filtered \
  --processes $BOOTLEG_PROCESS_COUNT

python3 -m bootleg_data_prep.prep_generate_slices_part2 \
  --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
  --subfolder_name kore50_aliases_filtered \
  --processes $BOOTLEG_PROCESS_COUNT \
  --emb_dir $BOOTLEG_WIKIPEDIA_DIR/embs \
  --kg_adj kg_adj_1129.txt \
  --kg_triples kg_triples_1129.json \
  --hy_vocab '' \
  --hy_types '' \
  --wd_vocab wikidatatitle_to_typeid_1129.json \
  --wd_types wikidata_types_1129.json \
  --rel_vocab relation_to_typeid_1129.json \
  --rel_types kg_relation_types_1129.json \
  --overwrite_saved_entities

python3 -m bootleg_data_prep.generate_slices \
  --subfolder_name kore50_aliases_filtered \
  --data_dir $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump \
  --processes $BOOTLEG_PROCESS_COUNT \
  --emb_dir $BOOTLEG_WIKIPEDIA_DIR/embs \
  --kg_adj kg_adj_1129.txt \
  --kg_triples kg_triples_1129.json \
  --hy_vocab ''\
  --hy_types ''\
  --kg_vocab $BOOTLEG_CODE_DIR/bootleg_data_prep/utils/param_files/pid_names_$BOOTLEG_LANG_CODE.json \
  --wd_vocab wikidatatitle_to_typeid_1129.json \
  --wd_types wikidata_types_1129.json \
  --rel_vocab relation_to_typeid_1129.json \
  --rel_types kg_relation_types_1129.json

cp -r $BOOTLEG_WIKIPEDIA_DIR/data/wiki_dump/kore50_aliases_filtered_0_-1_final $BOOTLEG_OUTPUT_DIR
