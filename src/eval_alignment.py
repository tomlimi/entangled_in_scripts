import os
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
from transformers import DataCollatorWithPadding
import argparse
import logging
import torch
from scipy.optimize import linear_sum_assignment
import json

from alignment_dataset import TatoebaAlignmentDataset, XtremeTatoebaAlignmentDataset
from utils import load_config


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    

def align_tensors(src_lang_tensors: torch.tensor, tgt_lang_tensors: torch.tensor) -> torch.tensor:
    """Aligns two tensors of language embeddings (src_lang_tensors and tgt_lang_tensors) using the Hungarian algorithm.
    Returns the alignment matrix and the alignment score.
    """
    # compute cosine similarity between tensors
    src_lang_tensors = src_lang_tensors / torch.norm(src_lang_tensors, dim=1, keepdim=True)
    tgt_lang_tensors = tgt_lang_tensors / torch.norm(tgt_lang_tensors, dim=1, keepdim=True)
    similarity_matrix = torch.mm(src_lang_tensors, tgt_lang_tensors.t())
    
    # Compute the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    
    alignment_matrix = torch.zeros(src_lang_tensors.size(0), tgt_lang_tensors.size(0))
    alignment_matrix[row_ind, col_ind] = 1
    return alignment_matrix
    
    
def eval_metric(inputs, model: XLMRobertaModel,metric_name: str) -> float:
    
    src_input_ids = torch.LongTensor(inputs['src_input_ids'])
    tgt_input_ids = torch.LongTensor(inputs['tgt_input_ids'])
    
    src_attention_mask = torch.FloatTensor(inputs['src_attention_mask'])
    tgt_attention_mask = torch.FloatTensor(inputs['tgt_attention_mask'])
    
    with torch.no_grad():
        src_lang_tensors = mean_pooling(model(input_ids=src_input_ids,
                                              attention_mask=src_attention_mask), src_attention_mask)
        tgt_lang_tensors = mean_pooling(model(input_ids=tgt_input_ids,
                                              attention_mask=tgt_attention_mask), tgt_attention_mask)
        
    alignment_matrix = align_tensors(src_lang_tensors, tgt_lang_tensors)
    
    if metric_name == 'accuracy':
        matches = torch.sum(torch.diagonal(alignment_matrix))
        result = matches / alignment_matrix.size(0)
    else:
        raise ValueError(f"Metric {metric_name} not implemented")
    return result.item()


def evaluate(args):
    
    logging.info('Evaluating Alignment (unsupervised)')

    seed = args.seed
    overwrite = args.overwrite
    
    src_lang = args.language_src
    tgt_lang = args.language_tgt
    metric_name = args.metric

    pt_in_path = os.path.join(args.pretrain_path, args.pretrain_name + '_' +str(seed))
    output_path = os.path.join(args.out_path, args.pretrain_name + '_' + str(seed), src_lang,
                            metric_name + '_evaluation', tgt_lang)
    
    logging.info(f"pt in: {pt_in_path}")
    logging.info(f"out: {output_path}")

    model_config = load_config(args.model_config_path)

    logging.info(f"Output is: {output_path}")
    if os.path.exists(os.path.join(output_path, f'{metric_name}_all.txt')) and not overwrite:
        logging.info(
            f"stats already exist at {os.path.join(output_path, f'{metric_name}_all.txt')}, no overrite. finishing.")
        return

    logging.info(f"Loading tokenizer")
    if 'tokenizer_lang' in model_config:
        language_list = model_config['tokenizer_lang']
        lang_index_src = language_list.index(src_lang)
        lang_index_tgt = language_list.index(tgt_lang)
    
        tokenizer_src = XLMRobertaTokenizerFast.from_pretrained(model_config['tokenizer_path'][lang_index_src],
                                                            max_length=model_config['max_sent_len'])
        tokenizer_tgt = XLMRobertaTokenizerFast.from_pretrained(model_config['tokenizer_path'][lang_index_tgt],
                                                            max_length=model_config['max_sent_len'])
        lang_offset_src = len(tokenizer_src) * lang_index_src
        lang_offset_tgt = len(tokenizer_tgt) * lang_index_tgt
    else:
        tokenizer_src = XLMRobertaTokenizerFast.from_pretrained(model_config['tokenizer_path'],
                                                            max_length=model_config['max_sent_len'])
        tokenizer_tgt = None
        
        lang_offset_src = 0
        lang_offset_tgt = 0
        
    logging.info("Loading dataset...")

    if src_lang == 'en' or tgt_lang == 'en':
        dataset = XtremeTatoebaAlignmentDataset(src_lang, tgt_lang, tokenizer_src, tokenizer_tgt,
                                                max_length=model_config['max_sent_len'],
                                                lang_offset_src=lang_offset_src, lang_offset_tgt=lang_offset_tgt)
    else:
        dataset = TatoebaAlignmentDataset(src_lang, tgt_lang, tokenizer_src, tokenizer_tgt,
                                          max_length=model_config['max_sent_len'],
                                          lang_offset_src=lang_offset_src, lang_offset_tgt=lang_offset_tgt)

    logging.info(f"Loading model")
    model = XLMRobertaModel.from_pretrained(pt_in_path)
    logging.info(f"#params:, {model.num_parameters()}")

    logging.info("Gathering stats...")

    result = eval_metric(dataset.test, model, metric_name)

    logging.info(f"Model {metric_name}: {result}")
    # saving the stats:

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'{metric_name}_all.txt'), 'w') as eval_out:
        json.dump({f'eval_{metric_name}': result}, eval_out)
    with open(os.path.join(output_path, f'script_args.txt'), 'w') as script_args:
        json.dump({'args': str(vars(args)), 'note': 'overrite is manually always True.'}, script_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', type=str, required=True)
    parser.add_argument('-i', '--pretrain_path', type=str, required=True)
    parser.add_argument('-p', '--pretrain_name', type=str, required=True)
    parser.add_argument('-s', '--language_src', type=str, required=True)
    parser.add_argument('-t', '--language_tgt', type=str, required=True)
    parser.add_argument('--model_config_path',type=str, required=True)
    parser.add_argument('--metric', type=str, default='accuracy', required=False)
    parser.add_argument('--overwrite', type=bool, default=True)
    parser.add_argument('--seed',type=int, required=False, default=1234)
    args = parser.parse_args()
    logging.info(vars(args))
    evaluate(args)