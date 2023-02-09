import json
import os
from collections import OrderedDict

from notebooks.notebook_utils import distribution_from_frequencies


def load_config(config_path):
    with open(config_path, 'r') as fp:
        return json.load(fp)
    
    
def get_distributions_over_decoded_vocabulary_default(result_dir: str, languages: list[str]) -> (OrderedDict, OrderedDict):
    """
    Get the distribution over the vocabulary for each language. For given tokenizer.
    """
    
    frequencies_over_vocabulary = {}
    for lang in languages:
        tokenizer_stats_path = os.path.join(result_dir, f"token_freq_{lang}_decoded.json")
        
        try:
            frequencies_over_vocabulary[lang] = json.load(open(tokenizer_stats_path, 'r'))
        except FileNotFoundError:
            print(f"{lang} freq file not found ({tokenizer_stats_path}).")
            continue
            
    # multilingual frequency file
    tokenizer_stats_path = os.path.join(result_dir, f"token_frequencies_decoded.json")
    
    try:
        frequencies_over_vocabulary["All"] = json.load(open(tokenizer_stats_path, 'r'))
    except FileNotFoundError:
        print(f"Multilingual freq file not found ({tokenizer_stats_path}).")
    
    distribution_over_vocabulary = {}
    for lang, freqs in frequencies_over_vocabulary.items():
        freqs = OrderedDict([(tok, freq) for tok, freq in sorted(freqs.items(), key=lambda item: item[0])])
        distribution_over_vocabulary[lang] = distribution_from_frequencies(freqs)
        frequencies_over_vocabulary[lang] = freqs
    
    return distribution_over_vocabulary, frequencies_over_vocabulary
