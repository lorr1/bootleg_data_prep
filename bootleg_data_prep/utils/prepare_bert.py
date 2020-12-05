import os

import argh
import torch
from transformers import BertTokenizer, BertModel


@argh.arg('do_lower_case', type=bool, help='True for an uncased model and False for a cased model.')
def prepare_bert_models(cache_dir, transformers_model_name, do_lower_case: bool):
    """
    Fetch a model from the transformers model respository and save to a cache folder.
    Example:
    cache_dir = "pubmed_bert_models"
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    do_lower_case = True
    """
    case_mark = 'uncased' if do_lower_case else 'cased'
    print(f'Saving **{case_mark}** models to {cache_dir} from {model_name}')
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, do_lower_case=do_lower_case)
    torch.save(tokenizer, os.path.join(cache_dir, f"bert_base_{case_mark}_tokenizer.pt"))
    model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
    torch.save(model.encoder, os.path.join(cache_dir, f"bert_base_{case_mark}_encoder.pt"))
    torch.save(model.embeddings, os.path.join(cache_dir, f"bert_base_{case_mark}_embedding.pt"))
    if do_lower_case:
        print('Remember to set `word_embedding.use_lower_case` in config to True')


if __name__ == '__main__':
    argh.dispatch_command(prepare_bert_models)
