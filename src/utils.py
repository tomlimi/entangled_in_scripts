import json
from transformers import XLMRobertaTokenizerFast


def load_config(config_path):
    with open(config_path, "r") as fp:
        return json.load(fp)


def get_tokenizer(model_config):
    if "tokenizer_lang" in model_config:
        language_list = model_config["tokenizer_lang"]
        lang_index = language_list.index(language)

        tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            model_config["tokenizer_path"][lang_index],
            max_length=model_config["max_sent_len"],
        )
        lang_offset = len(tokenizer) * lang_index
    else:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            model_config["tokenizer_path"], max_length=model_config["max_sent_len"]
        )
        lang_offset = 0

    return tokenizer, lang_offset
