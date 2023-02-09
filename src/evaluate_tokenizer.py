"""
Script for evaluating the tokenizer properties.

Specify:
- data_list (listed paths to data)
- languages (languages of the data for each data path)
- tokenizer_name (HF tokenizer name or name of the tokenizer pre-saved in the output directory)
- output_dir (path to output directory to save the results)
- unk_token (optional, the unkonwn token in the vocabulary, by default `<unk>`)

Example:
python evaluate_tokenizer.py \
    --data_list data_en.txt data_en2.txt data_es.txt data_pl.txt \
    --languages en en es pl \
    --tokenizer_name xlm-roberta-base \
    --output_dir /home/tokenizers_evaluation \
    [--unk_token <unk>]
    
Outputs tokenizer properties in a json file ``  in the output directory.
- Overlap (in JSD)
- Vocab Allocation (Average Rank, Characters per Token)
- Coverage (1 - percentage of unknown tokens)

"""

import argparse
import json
import os
from os import path
import numpy as np
from scipy.spatial.distance import jensenshannon
from transformers import AutoTokenizer
import logging
from collections import defaultdict
from itertools import combinations

from compute_token_frequency import compute_frequencies
from utils import get_distributions_over_decoded_vocabulary_default


logging.basicConfig(level=logging.INFO)

UNK_TOKEN = "<unk>"


def compute_number_of_characters(lang2data: dict[str, list[str]]) -> dict[str, int]:
    
    number_of_characters = defaultdict(int)
    
    for lang, data_paths in lang2data.items():
        for data_path in data_paths:
            with open(data_path, "r") as data_f:
                for line in data_f:
                    number_of_characters[lang] += len(line)
                    number_of_characters["All"] += len(line)
                
    return number_of_characters


def compute_jsd(probabilities_l1, probabilities_l2, base=2.):
    return jensenshannon(probabilities_l1, probabilities_l2, base=base) ** 2


def compute_average_rank(probabilities):
    sorted_probabilities = np.sort(probabilities)[::-1]
    r_e = np.sum(sorted_probabilities * np.arange(len(probabilities)))
    return r_e


def get_properties(languages, out_dir, number_of_characters, unk_token=UNK_TOKEN):
    
    vocab_distributions, vocab_frequencies = \
        get_distributions_over_decoded_vocabulary_default(out_dir, languages)
    
    
    languages = list(set(languages)) + ["All"]
    vocab_distributions_arr = {lang: np.array(list(vocab_distributions[lang].values())) for lang in languages}
    properties = {}
    
    # computes Overalp (in JSD)
    logging.info("Computing Overlap (JSD)...")
    properties['JSD'] = {}
    for lang1, lang2 in combinations(languages, 2):
        properties['JSD'][f'{lang1}-{lang2}'] = \
            compute_jsd(vocab_distributions_arr[lang1], vocab_distributions_arr[lang2])
        
    # computes Vocab Allocation (Average Rank)
    logging.info("Computing Vocab Allocation (Average Rank)...")
    properties['Average Rank'] = {}
    for lang in languages:
        properties['Average Rank'][lang] = \
            compute_average_rank(vocab_distributions_arr[lang])
            
    number_of_tokens = {lang: np.sum(list(vocab_frequencies[lang].values())) for lang in languages}
    
    # compute (Characters per Token)
    logging.info("Computing Characters per Token...")
    properties['Characters per Token'] = {}
    for lang in languages:
        properties['Characters per Token'][lang] = number_of_characters[lang] / number_of_tokens[lang]
        
    # compute Coverage (percentage of unknown tokens)
    logging.info("Computing Coverage (1 - percentage of unknown tokens)...")
    properties['Coverage'] = {}
    for lang in languages:
        if unk_token in vocab_frequencies[lang]:
            properties['Coverage'][lang] = 1. - (vocab_frequencies[lang][unk_token] / number_of_tokens[lang])
        else:
            logging.warning(f"Unknown token {unk_token} not in vocabulary for language {lang}.")
            properties['Coverage'][lang] = None
            
    return properties


def main(args):
    
    tokenizer_path = os.path.join(args.output_path, args.tokenizer_name)
    logging.info(f"Looking for tokenizer at {tokenizer_path}")
    
    if not os.path.exists(tokenizer_path):
        logging.info("Tokenizer not found. Downloading tokenizer from HF.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        logging.info(f"Saving tokenizer to {tokenizer_path}")
        tokenizer.save_pretrained(tokenizer_path)
    
    lang2data = defaultdict(list)
    for lang, data_path in zip(args.languages, args.data_list):
        lang2data[lang].append(data_path)

    for lang, data_paths in lang2data.items():
        if not path.exists(os.path.join(tokenizer_path, f"token_freq_{lang}_decoded.json")):
            logging.info(f"Computing token frequencies for {lang}")
            compute_frequencies(args.languages, data_paths, tokenizer_path, name=f'token_freq_{lang}')
    if not path.exists(os.path.join(tokenizer_path, "token_frequencies_decoded.json")):
        compute_frequencies(args.languages, args.data_list, tokenizer_path)
    
    number_of_characters = compute_number_of_characters(lang2data)

    t_properties = get_properties(args.languages, tokenizer_path, number_of_characters, args.unk_token)

    output_path = os.path.join(tokenizer_path, "tokenizer_properties.json")
    # save results
    logging.info(f"Saving tokenizer properties to {output_path}")
    with open(output_path, "w") as f:
        json.dump(t_properties, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_list", nargs="+", help="<Required> Set flag", required=True
    )
    parser.add_argument(
        "-l", "--languages", nargs="+", help="List of languages the tokenizer was trained on.", required=True,
    )
    parser.add_argument(
        "-t", "--tokenizer_name", type=str, help="Path to tokenizer", required=True
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="Path to output file", required=True
    )
    parser.add_argument(
        "-u", "--unk_token", type=str, help="UNK token", required=False, default=UNK_TOKEN
    )
    
    args = parser.parse_args()
    main(args)
