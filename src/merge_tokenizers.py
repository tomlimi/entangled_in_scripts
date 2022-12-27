import argparse
import json
import logging
import numpy as np
from collections import defaultdict
import os
from notebooks.notebook_utils import get_tokenizer, get_word_logits, substitute_word_logits


def merge_vocabularies_with_logits(token_logit_dict, NV, weights_dict=None):
    
    merged_logits = defaultdict(float)
    for lang, tl in token_logit_dict.items():
        if weights_dict is not None:
            weight = weights_dict[lang]
        else:
            weight = 1 / len(token_logit_dict)
            
        for token, logit in tl.items():
            merged_logits[token] += np.exp(logit) * weight
    
    merged_vocabulary = dict(sorted(merged_logits.items(), key=lambda item: -item[1])[:NV])
    
    norm_sum = sum([v for t, v in merged_vocabulary.items() if t not in('<s>', '<pad>', '</s>',  '<unk>', '<mask>')])
    merged_vocabulary = {t: np.log(v/norm_sum) if t not in('<s>', '<pad>', '</s>',  '<unk>', '<mask>') else 0.0
                         for t, v in merged_vocabulary.items()}
    return merged_vocabulary


def merge_tokenizers(tokenizer_dir, languages, vocab_size_mono, vocab_size_merged, alpha, type):
    """ Function merging tokenizers for different languages into one """
    suffix = 'merged'

    mono_word_logits = {lang: get_word_logits(tokenizer_dir, type, lang,alpha, vocab_size_mono) for lang in languages}
    word_logit_dict = merge_vocabularies_with_logits(mono_word_logits, vocab_size_merged)

    hyphenated_languages = '-'.join(languages)
    
    substitute_word_logits(tokenizer_dir, type, hyphenated_languages, alpha, vocab_size_merged,
                           word_logit_dict, suffix)
    
    multi_tokenizer = get_tokenizer(tokenizer_dir, type , hyphenated_languages, alpha, vocab_size_merged)
    
    out_path = os.path.join(tokenizer_dir, f"{type}-{suffix}", hyphenated_languages, f"alpha-{alpha}_N-{vocab_size_merged}")
    logging.info(f"Saving tokenizer to {out_path}")
    
    multi_tokenizer = get_tokenizer(tokenizer_dir, f"{type}-{suffix}", hyphenated_languages, alpha, vocab_size_merged)
    multi_tokenizer.save_pretrained(out_path)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Directory to save tokenizers")
    parser.add_argument("--languages", type=str, required=True, nargs='+', help="Languages to merge tokenizers for.")
    parser.add_argument("--type", type=str, default="sp-unigram", help="Type of tokenizer")
    parser.add_argument("--vocab_size_mono", type=int, default=40000, help="Vocab size")
    parser.add_argument("--vocab_size_merged", type=int, default=120000, help="Vocab size")
    parser.add_argument("--alpha", type=str, default="0.25", help="Alpha for merging")
    args = parser.parse_args()
    
    merge_tokenizers(args.tokenizer_dir, args.languages, args.vocab_size_mono, args.vocab_size_merged, args.alpha, args.type)