import torch
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast, set_seed
import os
import json
import numpy as np
import argparse
from mlm_dataset import LineByLineTextDataset, DataCollatorForLanguageModeling
import logging
from transformers import AutoModelForMaskedLM

logging.basicConfig(level=logging.INFO)
logging.info(torch.cuda.is_available())
set_seed(10)


def compute_metrics(p):
    logits = p.predictions
    labels = p.label_ids

    masked_indices = labels != -100
    masked_words_num = np.count_nonzero(masked_indices)
    preds_logits = logits[masked_indices]

    labels_at_mask = labels[masked_indices].reshape(-1, 1)

    logits_at_mask = np.take_along_axis(preds_logits, labels_at_mask, axis=1)
    correct_word_indices = np.sum(preds_logits > logits_at_mask, axis=1)

    mrr = (np.sum(1 / (correct_word_indices + 1)) / masked_words_num)

    return {'mrr': mrr}


def compute_mrr(model, ft_eval, data_collator, vocab_size, batch_size):

    mrrs = []
    rank_accs = []
    num_of_masked = []
    for idx in tqdm(range(0,len(ft_eval), batch_size), desc='getting eval results...'):
        data_dict = data_collator([ft_eval[i]['input_ids'] for i in range(min(batch_size, len(ft_eval)-idx))]) #dict: {'input_ids':<tokens>, 'labels':<-100 should be ignored>}
        inputs, labels = data_collator.mask_tokens(data_dict['input_ids'])
        logits = model(inputs).logits
        masked_indices = labels != -100
        masked_words_num = np.count_nonzero(masked_indices)
        preds_logits = logits[masked_indices].detach().numpy()
        num_of_masked.append(masked_words_num)
        labels_at_mask = labels[masked_indices].numpy().reshape(-1, 1)

        logits_at_mask = np.take_along_axis(preds_logits, labels_at_mask, axis=1)
        correct_word_indices = np.sum(preds_logits > logits_at_mask, axis=1)

        mrrs.append((np.sum(1/(correct_word_indices+1))/masked_words_num))
        rank_accs.append(np.sum(1-(correct_word_indices/vocab_size))/len(correct_word_indices))

    return mrrs, rank_accs, num_of_masked


def eval_single_model(args):
    """
    runs a single evaluation that produces an average MRR score over the sentences in the input files. The MRR score is
    saved in the specified path.
    :param args: reffer to the 'help' arguments in the argparse section below.
    """
    # initializing variables w.r.t args:
    model_dir_path = args.model_dir_path
    model_config_path = args.model_config_path
    eval_data_paths = args.eval_data_paths
    truncate_at = args.truncate_at

    overwrite = args.overwrite
    is_zero_shot = args.is_zero_shot
    out_path = args.out_path if args.out_path is not None else model_dir_path
    base_name = 'mrr_eval_' if not is_zero_shot else 'mrr_eval_zero_shot'
    logging.info(f"Output is: {out_path} with base name: {base_name}")
    config =json.load(open(model_config_path,'r'))

    language = args.language
    
    if 'tokenizer_lang' in config:
        language_list = config['tokenizer_lang']
        lang_index = language_list.index(language)
        
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(config['tokenizer_path'][lang_index])
        vocab_size=len(tokenizer) * len(language_list)
        lang_to_offset = {language: len(tokenizer) * lang_index}
    else:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(config['tokenizer_path'])
        vocab_size=len(tokenizer)
        lang_to_offset = {}

    eval_lang_paths = [(language, path) for path in eval_data_paths]
    
    if truncate_at >= 1:
        if os.path.exists(os.path.join(out_path,f'{base_name+str(truncate_at)}.txt')) and not overwrite:
            logging.info(f"stats already exist at {os.path.join(out_path,f'{base_name+str(truncate_at)}.txt')}, no overwrite. returning. ")
            return
    if truncate_at < 1:
        if os.path.exists(os.path.join(out_path,f'{base_name}all.txt')) and not overwrite:
            logging.info(f"stats already exist at {os.path.join(out_path,f'{base_name+str(truncate_at)}.txt')}, no overwrite. returning.")
            return
    logging.info("Gathering stats...")

    model = AutoModelForMaskedLM.from_pretrained(model_dir_path)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, vocab_size=vocab_size, mlm=True, mlm_probability=0.15
    )
    
    ft_eval = LineByLineTextDataset({language: tokenizer}, lang_paths=eval_lang_paths, block_size=config['max_sent_len'], truncate_at=truncate_at, lang_to_offset=lang_to_offset, randomize=False)

    logging.info(f"Evaulating {model_dir_path} on {eval_data_paths} with truncate {truncate_at} and zeroshot {is_zero_shot}. Overrite:{overwrite}.")
    # gathering scores:
    batch_size = 16

    mrrs, rank_accs, num_of_masked = compute_mrr(model, ft_eval, data_collator, config['vocab_size'], batch_size)

    mrr = np.sum(np.array(mrrs) * np.array(num_of_masked)) / sum(num_of_masked)
    rank_acc = np.sum(np.array(rank_accs) * np.array(num_of_masked)) / sum(num_of_masked)

    logging.info(f"Model mrr: {mrr}")
    logging.info(f"Model rank accuracy: {rank_acc}")
    # saving the stats:
    if truncate_at < 1:
        truncate_at = "all"
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path,f'{base_name+str(truncate_at)}.txt'), 'w') as eval_mrr_out:
        json.dump({'eval_mrr':mrr, 'all_batches_mrrs':mrrs, 'num_of_masked_per_batch':num_of_masked}, eval_mrr_out)
    with open(os.path.join(out_path,f'rank_acc_eval_{base_name+str(truncate_at)}.txt'), 'w') as eval_rank_acc_out:
        json.dump({'eval_rank_acc':rank_acc, 'all_batches_rank_acc':rank_accs}, eval_rank_acc_out)
    with open(os.path.join(out_path,f'script_args_{base_name+str(truncate_at)}.txt'), 'w') as script_args:
        json.dump({'args':str(vars(args)),'note':'overwrite is manually always True.'}, script_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval_data_paths', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-m', '--model_dir_path', type=str, help='<Required> Set flag', required=True)
    parser.add_argument('-c', '--model_config_path',type=str, required=True)
    parser.add_argument('-l', '--language', type=str, required=True)
    parser.add_argument('-t', '--truncate_at',type=int, default=-1)
    parser.add_argument('--overwrite',type=bool, default=True)
    parser.add_argument('-z', '--is_zero_shot',type=bool, default=False)
    parser.add_argument('-o', '--out_path',type=str, default=None)
    parser.add_argument('-r', '--rand_model',type=bool, default=False)
    args = parser.parse_args()
    logging.info(vars(args))
    eval_single_model(args)

#    --eval_data_paths /cs/snapless/gabis/danmh/language-graph/new_data/en/10mb_data/test.txt --model_dir_path /cs/snapless/gabis/danmh/multi-teacher/model/LMs/en_de/en_de/en_de_10 --model_config_path /cs/snapless/gabis/danmh/multi-teacher/model/LMs/en_de/en_de_student_model.json --truncate_at 300 --out_path /cs/snapless/gabis/danmh/multi-teacher/model/LMs/en_de/en_de/en_de_10/eval_en

#    --eval_data_paths /cs/snapless/gabis/danmh/language-graph/new_data/en/10mb_data/test.txt --model_dir_path /cs/snapless/gabis/danmh/multi-teacher/model/LMs/en_he/en_he/en_he_10 --model_config_path /cs/snapless/gabis/danmh/multi-teacher/model/LMs/en_he/en_he_student_model.json --truncate_at 300 --out_path /cs/snapless/gabis/danmh/multi-teacher/model/LMs/en_he/en_he/en_he_10/eval_en

