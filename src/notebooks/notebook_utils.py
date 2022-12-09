import os
import json
import csv
from transformers import XLMRobertaTokenizerFast
import logging
import pandas as pd


def get_tokenizer_path(tokenizer_dir, tokenizer_type, lang, alpha, NV):
    return os.path.join(tokenizer_dir, tokenizer_type, lang, f"alpha-{alpha}_N-{NV}")


# getting tokenizer
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
