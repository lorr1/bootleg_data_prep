https://github.com/facebookresearch/KILT
Data taken and processed through https://github.com/facebookresearch/KILT/blob/master/scripts/create_kilt_data_paragraphs.py up until the last step

This script takes that data, and generates the jsonl files used for merge_shuff_split and the rest of the data prep pipeline so we can
generate slices with the data.

The order of running is


- generate_aliases: will iterate over the data and (1) transform aliases links to the QIDs, (2) generate alias to qids, and (3) merge this with Wikidata aliases. Will output new data and entity_db.
- filter_data: remove mentions we don't have matched for in top 30 candidates
- process_data: will get the data from the previous step into the correct format with word offsets, qids, anchors, and sentences.