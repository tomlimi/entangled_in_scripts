import os
import numpy as np
from collections import defaultdict
import json
import csv
from transformers import XLMRobertaTokenizerFast
import logging
import pandas as pd
from collections import OrderedDict

TOKENIZERS_DIR = "/home/limisiewicz/my-luster/entangled-in-scripts/tokenizers"
MODELS_DIR = "/home/limisiewicz/my-luster/entangled-in-scripts/models"

def get_tokenizer_path(tokenizer_dir, tokenizer_type, lang, alpha, NV):
    return os.path.join(tokenizer_dir, tokenizer_type, lang, f"alpha-{alpha}_N-{NV}")


# getting tokenizer / vocabularies
def get_tokenizer(tokenizer_dir, tokenizer_type, lang, alpha, NV):
    tokenizer_path = get_tokenizer_path(tokenizer_dir, tokenizer_type, lang, alpha, NV)
    return XLMRobertaTokenizerFast.from_pretrained(tokenizer_path, unk_token="<unk>")


def get_token_frequencies(tokenizer_dir, tokenizer_type, lang, alpha, NV):
    tokenizer_path = get_tokenizer_path(tokenizer_dir, tokenizer_type, lang, alpha, NV)
    with open(os.path.join(tokenizer_path, "decoded_token_frequencies.json")) as f:
        frequencies = json.load(f)
    return frequencies


def get_token_stats(tokenizer_dir, tokenizer_type, languages, alphas, NVs):
    token_stats = {}
    for alpha in alphas:
        token_stats[alpha] = {}
        for lang, NV in zip(languages, NVs):
            token_stats[alpha][lang] = get_token_frequencies(
                tokenizer_dir, tokenizer_type, lang, alpha, NV
            )
    return token_stats


def get_tokenizers(tokenizer_dir, tokenizer_type, languages, alphas, NVs):
    tokenizers = {}
    for alpha in alphas:
        tokenizers[alpha] = {}
        for lang, NV in zip(languages, NVs):
            tokenizers[alpha][lang] = get_tokenizer(
                tokenizer_dir, tokenizer_type, lang, alpha, NV
            )
    return tokenizers


def compute_char_stats(token_stats, char_fn, skip_tokens):
    char_stats = {}
    for token, freq in token_stats.items():
        if token in skip_tokens:
            continue
        for char in token:
            mapped_char = char_fn(char)
            if mapped_char not in char_stats:
                char_stats[mapped_char] = 0
            char_stats[mapped_char] += freq
    return char_stats


def get_char_stats(token_stats, languages, alphas, char_fn, skip_tokens):
    char_stats = {}
    for alpha in alphas:
        char_stats[alpha] = {}
        for lang in languages:
            char_stats[alpha][lang] = compute_char_stats(
                token_stats[alpha][lang], char_fn, skip_tokens
            )
    return char_stats


class UnicodeBlocks:
    def __init__(self, blocks_tsv_path):
        self.blocks = []
        with open(blocks_tsv_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # convert first and second row from hex string to int
                self.blocks.append((int(row[0], 16), int(row[1], 16), row[2]))

    def get_block(self, char):
        char_int = ord(char)
        for block in self.blocks:
            if char_int >= block[0] and char_int <= block[1]:
                return block[2]
        # raise ValueError(f"Character {char} not in any block")
        logging.warning(f"Character {char} (unicode '{ord(char)}') not in any block")


def get_alphabet_occurence(token_stats):
    def _is_alphabet(token):
        return len(token) == 1

    alphabet_occurence = 0
    for token, freq in token_stats.items():
        if _is_alphabet(token):
            alphabet_occurence += freq

    return alphabet_occurence


def get_total_occurence(token_stats):
    return sum(token_stats.values())


def apply_to_all_token_stats(token_stats, fn):
    return {
        alpha: {lang: fn(token_stats[alpha][lang]) for lang in token_stats[alpha]}
        for alpha in token_stats
    }


def stats_to_pandas(token_stats):
    df = pd.DataFrame(
        [
            {
                "alpha": alpha,
                "language": lang,
                "value": token_stats[alpha][lang],
            }
            for alpha in token_stats
            for lang in token_stats[alpha]
        ]
    )
    df = df.set_index(["alpha", "language"])
    return df


def get_word_logits(tokenizer_dir, tokenizer_type, lang, alpha, NV):
    """
    Returns word logits from tokenizer.json.
    """
    tokenizer_path = os.path.join(tokenizer_dir, tokenizer_type, lang, f"alpha-{alpha}_N-{NV}", "tokenizer.json")
    with open(tokenizer_path, 'r') as tokenizer_json:
        tokenizer_dict = json.load(tokenizer_json)
    return {word_logit[0]: word_logit[1] for word_logit in tokenizer_dict['model']['vocab']}


def substitute_word_logits(tokenizer_dir, tokenizer_type, lang, alpha, NV, word_logits, suffix='merged'):
    """
    Substitutes word logits in tokenizer.json with the ones provided.
    """
    tokenizer_in_path = os.path.join(tokenizer_dir, tokenizer_type, lang, f"alpha-{alpha}_N-{NV}", "tokenizer.json")
    with open(tokenizer_in_path, 'r') as tokenizer_json:
        tokenizer_dict = json.load(tokenizer_json)

    tokenizer_dict['model']['vocab'] = [[t, v] for t, v in word_logits.items()]
    
    tokenizer_out_path = os.path.join(tokenizer_dir, f"{tokenizer_type}-{suffix}", lang, f"alpha-{alpha}_N-{NV}", "tokenizer.json")
    os.makedirs(os.path.dirname(tokenizer_out_path), exist_ok=True)
    with open(tokenizer_out_path, 'w', encoding='utf-8' ) as tokenizer_json:
        json.dump(tokenizer_dict, tokenizer_json, indent=2, ensure_ascii=False)


# alphabet analysis
def get_alphabet_size(tokenizer):
    return len([t for t in tokenizer.get_vocab().keys() if len(t) == 1])


def print_alphabet_stats(tokenizer, tokenizer_name=None, NV=None):
    if NV is None:
        NV = len(tokenizer.get_vocab())
    NA = get_alphabet_size(tokenizer)

    if tokenizer_name:
        print("Results for XLM-Roberta.")
    print(f"Size of alphabet: {NA}")
    print(f"Size of vocabulary: {NV}")
    print(f"Ratio: {NA/NV}")


# tokens acceptance (language representation)
def tokens_acceptance(mono_tokenizer, multi_tokenizer):
    n_overlap = len(
        set.intersection(
            set(mono_tokenizer.get_vocab().keys()),
            set(multi_tokenizer.get_vocab().keys()),
        )
    )
    return n_overlap / len(mono_tokenizer.get_vocab())


def tokens_acceptance_no_alphabet(mono_tokenizer, multi_tokenizer):
    n_overlap = len(
        set.intersection(
            set([k for k in mono_tokenizer.get_vocab().keys() if len(k) > 1]),
            set([k for k in multi_tokenizer.get_vocab().keys() if len(k) > 1]),
        )
    )
    return n_overlap / len([k for k in mono_tokenizer.get_vocab().keys() if len(k) > 1])


# token overlaps
def tokens_overlap(mono_tokenizer_list, multi_tokenizer):

    all_mono_vocabs = (
        set(mono_tokenizer.get_vocab().keys()) for mono_tokenizer in mono_tokenizer_list
    )

    all_mono_overlap = set.intersection(*all_mono_vocabs)
    n_all_mono_overlap = len(all_mono_overlap)
    n_all_overlap = len(
        set.intersection(all_mono_overlap, set(multi_tokenizer.get_vocab().keys()))
    )
    return n_all_mono_overlap / len(mono_tokenizer_list[0].get_vocab())


def tokens_overlap_exact(mono_tokenizer_list, multi_tokenizer):

    all_mono_vocabs = (
        set(mono_tokenizer.get_vocab().keys()) for mono_tokenizer in mono_tokenizer_list
    )

    all_mono_overlap = set.intersection(*all_mono_vocabs)
    n_all_mono_overlap = len(all_mono_overlap)
    n_all_overlap = len(
        set.intersection(all_mono_overlap, set(multi_tokenizer.get_vocab().keys()))
    )
    return n_all_overlap / len(mono_tokenizer_list[0].get_vocab())


def tokens_overlap_exact_no_alphabet(mono_tokenizer_list, multi_tokenizer):

    all_mono_vocabs = (
        set([k for k in mono_tokenizer.get_vocab().keys() if len(k) > 1])
        for mono_tokenizer in mono_tokenizer_list
    )

    all_mono_overlap = set.intersection(*all_mono_vocabs)
    n_all_mono_overlap = len(all_mono_overlap)
    n_all_overlap = len(
        set.intersection(
            all_mono_overlap,
            set([k for k in multi_tokenizer.get_vocab().keys() if len(k) > 1]),
        )
    )
    return n_all_overlap / len(
        [k for k in multi_tokenizer.get_vocab().keys() if len(k) > 1]
    )


def print_tokens_overlap(mono_tokenizer_list, multi_tokenizer):
    all_mono_vocabs = (
        set(mono_tokenizer.get_vocab().keys()) for mono_tokenizer in mono_tokenizer_list
    )
    sorted_tokens = sorted(
        set.intersection(*all_mono_vocabs, set(multi_tokenizer.get_vocab().keys()))
    )
    print(sorted_tokens)
    print(f"Number of overlapping tokens: {len(sorted_tokens)}")
    print("\n")


def merge_vocabularies_with_logits(token_logit_list, NV):
    merged_logits = defaultdict(float)
    for tl in token_logit_list:
        for token, logit in tl.items():
            merged_logits[token] += np.exp(logit)

    merged_vocabulary = dict(sorted(merged_logits.items(), key=lambda item: -item[1])[:NV])
    
    norm_sum = sum([v for t, v in merged_vocabulary.items() if t not in('<s>', '<pad>', '</s>',  '<unk>', '<mask>')])
    merged_vocabulary = {t: np.log(v/norm_sum) if t not in('<s>', '<pad>', '</s>',  '<unk>', '<mask>') else 0.0
                         for t, v in merged_vocabulary.items()}
    return merged_vocabulary


def distribution_from_frequencies(frequencies: dict) -> OrderedDict:

    sum_freq = np.sum(list(frequencies.values()))
    distribution = OrderedDict([(tok, freq / sum_freq) for tok, freq in frequencies.items()])
    return distribution


def get_distribution_over_vocabulary(tok_type: str, alpha: float, NV: int, languages: list[str]) -> (dict, dict):
    
    tok_type_map = {'multilingual': 'sp-unigram',
                    '20l-multilingual': 'sp-unigram',
                    'merged': 'sp-unigram-merged',
                    '20l-merged': 'sp-unigram-merged',
                    'nooverlap': 'sp-unigram',
                    'bpe': 'sp-bpe',
                    '20l-bpe': 'sp-bpe',
                    'bpe_nooverlap': 'sp-bpe'}
    
    frequencies_over_vocabulary = dict()
    
    # monolingual frequencies
    for lang in languages:
        if "nooverlap" in tok_type:
            tokenizer_stats_path = os.path.join(TOKENIZERS_DIR, tok_type_map[tok_type], lang,
                                                f"alpha-{alpha}_N-{NV//len(languages)}",
                                                f"token_freq_{lang}_{alpha}.json")
        else:
            tokenizer_stats_path = os.path.join(TOKENIZERS_DIR, tok_type_map[tok_type], '-'.join(languages),
                                                f"alpha-{alpha}_N-{NV}", f"token_freq_{lang}_{alpha}.json")
        try:
            frequencies_over_vocabulary[lang] = json.load(open(tokenizer_stats_path, 'r'))
        except FileNotFoundError:
            print(f"{lang} freq file not found ({tokenizer_stats_path}).")
            continue

    # multilingual frequency file
    if "nooverlap" in tok_type:
        frequencies_over_vocabulary_new = defaultdict(dict)
        
        for tok_idx in range(NV):
            lang_idx = tok_idx // (NV//len(languages))
            rel_tok_idx = str(tok_idx % (NV//len(languages)))
            curr_lang = languages[lang_idx]
            for lang in languages:
                frequencies_over_vocabulary_new[lang][str(tok_idx)] = frequencies_over_vocabulary[curr_lang][rel_tok_idx] if lang == curr_lang else 0
            frequencies_over_vocabulary_new['All'][str(tok_idx)] = frequencies_over_vocabulary[curr_lang][rel_tok_idx]
        frequencies_over_vocabulary = frequencies_over_vocabulary_new
    else:
        tokenizer_stats_path = os.path.join(TOKENIZERS_DIR, tok_type_map[tok_type], '-'.join(languages),
                                            f"alpha-{alpha}_N-{NV}", f"token_frequencies.json")
        try:
            frequencies_over_vocabulary['All'] = json.load(open(tokenizer_stats_path, 'r'))
        except FileNotFoundError:
            print(f"Multilingual freq file not found ({tokenizer_stats_path}).")
        
    distribution_over_vocabulary = dict()
    for lang, freqs in frequencies_over_vocabulary.items():
        freqs = OrderedDict([(str(tok), freqs[str(tok)]) for tok in range(NV)])
        distribution_over_vocabulary[lang] = distribution_from_frequencies(freqs)
        frequencies_over_vocabulary[lang] = freqs
    
    return distribution_over_vocabulary, frequencies_over_vocabulary
        

def get_mlm_results(tok_type: str, alpha: float, NV: int, languages: list[str],
                    seed=1234, alpha_train=0.25, metrics=('mrr', 'bpc')) -> dict:
    """
    tok_type: 'multilingual', 'merged', 'nooverlap', 'bpe', 'bpe-nooverlap'
    """
    # decreasing vocab size for nooverlap
    if 'nooverlap' in tok_type:
        NV = NV // len(languages)
        
    # load results
    results = {m: {} for m in metrics}
    for metric in metrics:
        for lang in languages:
            try:
                result_file = os.path.join(MODELS_DIR, "LM", f"{tok_type}-tokenization",
                                       f"alpha-{alpha}_alpha-train-{alpha_train}_N-{NV}_{seed}", lang,
                                       f"{metric}_eval_mrr_eval_all.txt")

                res = json.load(open(result_file, 'r'))[f"eval_{metric}"]
            except FileNotFoundError:
                try:
                    result_file = os.path.join(MODELS_DIR, "LM", f"{tok_type}-tokenization",
                                               f"alpha-{alpha}_alpha-train-{alpha_train}_N-{NV}_{seed}", lang,
                                               f"{metric}_eval_all.txt")
                    res = json.load(open(result_file, 'r'))[f"eval_{metric}"]
                except FileNotFoundError:
                    print(f"{result_file} not found.")
                    res = 0.0
            except json.JSONDecodeError:
                print(f"{result_file} couldn't be parsed.")
                res = 0.0
            results[metric][lang] = res

    return results


#TODO: write function to get results for downstream tasks
def get_downstream_results(tok_type: str, alpha: float, NV: int, languages: list[str], task: str,
                           ft_type='PROBE', seeds=(1234,1235, 1236, 1237, 1238), alpha_train=0.25, metrics=('f1-macro')) -> (dict, dict):
    
    """
    Get results for downstream tasks.
    """

    # decreasing vocab size for nooverlap
    if 'nooverlap' in tok_type:
        NV = NV // len(languages)
        
    results = {m: {} for m in metrics}
    results_avg = {m: {} for m in metrics}
    results_std = {m: {} for m in metrics}
    for metric in metrics:
        for src_lang in languages:
            results[metric][src_lang] = {}
            results_avg[metric][src_lang] = {}
            results_std[metric][src_lang] = {}
            for tgt_lang in languages:
                results[metric][src_lang][tgt_lang] = []
                for seed in seeds:
                    if not ft_type:
                        result_file = os.path.join(MODELS_DIR, task, f"{tok_type}-tokenization",
                                                   f"alpha-{alpha}_alpha-train-{alpha_train}_N-{NV}_{seed}",
                                                   src_lang, f"{metric}_evaluation", tgt_lang, f"{metric}_all.txt")
                    else:
                        result_file = os.path.join(MODELS_DIR, f"{task}_{ft_type}", f"{tok_type}-tokenization",
                                                   f"alpha-{alpha}_alpha-train-{alpha_train}_N-{NV}_{seed}",
                                                   src_lang, f"{metric}_evaluation", tgt_lang, f"{metric}_all.txt")
                    try:
                        results[metric][src_lang][tgt_lang].append(json.load(open(result_file, 'r'))[f"eval_{metric}"])
                    except FileNotFoundError:
                        print(f"{result_file} not found.")
                    except json.JSONDecodeError:
                        print(f"{result_file} couldn't be parsed.")
                        
                if len(results[metric][src_lang][tgt_lang]) > 0:
                    results_avg[metric][src_lang][tgt_lang] = np.mean(results[metric][src_lang][tgt_lang])
                    results_std[metric][src_lang][tgt_lang] = np.std(results[metric][src_lang][tgt_lang])
                else:
                    results_avg[metric][src_lang][tgt_lang] = 0.0
                    results_std[metric][src_lang][tgt_lang] = 0.0
                    
    return results_avg, results_std
    
        
corpus_sizes = {'en': {'alpha0.0': 30000000.0,
                       'alpha0.25': 122000000.0,
                       'alpha0.5': 496000000.0,
                       'alpha0.75': 2020000000.0,
                       'alpha1.0': 8200000000.0},
                'es': {'alpha0.0': 30000000.0,
                       'alpha0.25': 78400000.0,
                       'alpha0.5': 205000000.0,
                       'alpha0.75': 536000000.0,
                       'alpha1.0': 1400000000.0},
                'el': {'alpha0.0': 30000000.0,
                       'alpha0.25': 66900000.0,
                       'alpha0.5': 149000000.0,
                       'alpha0.75': 332000000.0,
                       'alpha1.0': 740000000.0},
                'zh': {'alpha0.0': 30000000.0,
                       'alpha0.25': 78400000.0,
                       'alpha0.5': 205000000.0,
                       'alpha0.75': 536000000.0,
                       'alpha1.0': 1400000000.0},
                'tr': {'alpha0.0': 30000000.0,
                       'alpha0.25': 61800000.0,
                       'alpha0.5': 127000000.0,
                       'alpha0.75': 262000000.0,
                       'alpha1.0': 540000000.0},
                'ar': {'alpha0.0': 30000000.0,
                       'alpha0.25': 61800000.0,
                       'alpha0.5': 127000000.0,
                       'alpha0.75': 262000000.0,
                       'alpha1.0': 540000000.0},
                'sw': {'alpha0.0': 30000000.0,
                       'alpha0.25': 30800000.0,
                       'alpha0.5': 31600000.0,
                       'alpha0.75': 32400000.0,
                       'alpha1.0': 33200000.0},
                'hi': {'alpha0.0': 30000000.0,
                       'alpha0.25': 51000000.0,
                       'alpha0.5': 86600000.0,
                       'alpha0.75': 147000000.0,
                       'alpha1.0': 250000000.0},
                'mr': {'alpha0.0': 30000000.0,
                       'alpha0.25': 30800000.0,
                       'alpha0.5': 31700000.0,
                       'alpha0.75': 32500000.0,
                       'alpha1.0': 33400000.0},
                'ur': {'alpha0.0': 30000000.0,
                       'alpha0.25': 39300000.0,
                       'alpha0.5': 51500000.0,
                       'alpha0.75': 67500000.0,
                       'alpha1.0': 88400000.0},
                'ta': {'alpha0.0': 30000000.0,
                       'alpha0.25': 43300000.0,
                       'alpha0.5': 62400000.0,
                       'alpha0.75': 90100000.0,
                       'alpha1.0': 130000000.0},
                'te': {'alpha0.0': 30000000.0,
                       'alpha0.25': 34700000.0,
                       'alpha0.5': 40100000.0,
                       'alpha0.75': 46400000.0,
                       'alpha1.0': 53600000.0},
                'th': {'alpha0.0': 30000000.0,
                       'alpha0.25': 69600000.0,
                       'alpha0.5': 162000000.0,
                       'alpha0.75': 375000000.0,
                       'alpha1.0': 870000000.0},
                'ru': {'alpha0.0': 30000000.0,
                       'alpha0.25': 106000000.0,
                       'alpha0.5': 371000000.0,
                       'alpha0.75': 1310000000.0,
                       'alpha1.0': 4600000000.0},
                'bg': {'alpha0.0': 30000000.0,
                       'alpha0.25': 70800000.0,
                       'alpha0.5': 167000000.0,
                       'alpha0.75': 394000000.0,
                       'alpha1.0': 930000000.0},
                'he': {'alpha0.0': 30000000.0,
                       'alpha0.25': 63700000.0,
                       'alpha0.5': 135000000.0,
                       'alpha0.75': 287000000.0,
                       'alpha1.0': 610000000.0},
                'ka': {'alpha0.0': 30000000.0,
                       'alpha0.25': 41500000.0,
                       'alpha0.5': 57400000.0,
                       'alpha0.75': 79500000.0,
                       'alpha1.0': 110000000.0},
                'vi': {'alpha0.0': 30000000.0,
                       'alpha0.25': 93200000.0,
                       'alpha0.5': 290000000.0,
                       'alpha0.75': 901000000.0,
                       'alpha1.0': 2800000000.0},
                'fr': {'alpha0.0': 30000000.0,
                       'alpha0.25': 78400000.0,
                       'alpha0.5': 205000000.0,
                       'alpha0.75': 536000000.0,
                       'alpha1.0': 1400000000.0},
                'de': {'alpha0.0': 30000000.0,
                       'alpha0.25': 83500000.0,
                       'alpha0.5': 232000000.0,
                       'alpha0.75': 647000000.0,
                       'alpha1.0': 1800000000.0}}

