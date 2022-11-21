import os
from transformers import XLMRobertaTokenizerFast


# getting tokenizer
def get_tokenizer(tokenizer_dir, tokenizer_type, lang, alpha, NV):
    tokenizer_path = os.path.join(tokenizer_dir, tokenizer_type, lang, f"alpha-{alpha}_N-{NV}")
    return XLMRobertaTokenizerFast.from_pretrained(tokenizer_path, unk_token = '<unk>')


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
    n_overlap = len(set.intersection(set(mono_tokenizer.get_vocab().keys()), set(multi_tokenizer.get_vocab().keys())))
    return n_overlap / len(mono_tokenizer.get_vocab())


def tokens_acceptance_no_alphabet(mono_tokenizer, multi_tokenizer):
    n_overlap = len(set.intersection(set([k for k in mono_tokenizer.get_vocab().keys() if len(k) > 1]),
                                     set([k for k in multi_tokenizer.get_vocab().keys() if len(k) > 1])))
    return n_overlap / len([k for k in mono_tokenizer.get_vocab().keys() if len(k) > 1])


# token overlaps
def tokens_overlap(mono_tokenizer_list, multi_tokenizer):
    
    all_mono_vocabs = (set(mono_tokenizer.get_vocab().keys()) for mono_tokenizer in mono_tokenizer_list)
    
    all_mono_overlap = set.intersection(*all_mono_vocabs)
    n_all_mono_overlap = len(all_mono_overlap)
    n_all_overlap = len(set.intersection(all_mono_overlap, set(multi_tokenizer.get_vocab().keys())))
    return n_all_mono_overlap / len(mono_tokenizer_list[0].get_vocab())


def tokens_overlap_exact(mono_tokenizer_list, multi_tokenizer):
    
    all_mono_vocabs = (set(mono_tokenizer.get_vocab().keys()) for mono_tokenizer in mono_tokenizer_list)
    
    all_mono_overlap = set.intersection(*all_mono_vocabs)
    n_all_mono_overlap = len(all_mono_overlap)
    n_all_overlap = len(set.intersection(all_mono_overlap, set(multi_tokenizer.get_vocab().keys())))
    return n_all_overlap / len(mono_tokenizer_list[0].get_vocab())


def tokens_overlap_exact_no_alphabet(mono_tokenizer_list, multi_tokenizer):
    
    all_mono_vocabs = (set([k for k in mono_tokenizer.get_vocab().keys() if len(k) > 1])
                       for mono_tokenizer in mono_tokenizer_list)
    
    all_mono_overlap = set.intersection(*all_mono_vocabs)
    n_all_mono_overlap = len(all_mono_overlap)
    n_all_overlap = len(set.intersection(all_mono_overlap,
                                         set([k for k in multi_tokenizer.get_vocab().keys() if len(k) > 1])))
    return n_all_overlap / len([k for k in multi_tokenizer.get_vocab().keys() if len(k) > 1])


def print_tokens_overlap(mono_tokenizer_list, multi_tokenizer):
    all_mono_vocabs = (set(mono_tokenizer.get_vocab().keys()) for mono_tokenizer in mono_tokenizer_list)
    sorted_tokens = sorted(set.intersection(*all_mono_vocabs, set(multi_tokenizer.get_vocab().keys())))
    print(sorted_tokens)
    print(f"Number of overlapping tokens: {len(sorted_tokens)}")
    print('\n')
