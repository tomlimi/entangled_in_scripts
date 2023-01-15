import torch
from abc import abstractmethod
from datasets import Dataset, load_dataset
import transformers


class UDDataset:
    lang_to_config = {
        "ar": "ar_nyuad",
        "el": "el_gdt",
        "en": "en_ewt",
        "es": "es_ancora",
        "tr": "tr_boun",
        "zh": "zh_gsd",
        "sw": "swl_sslc",
        "hi": "hi_hdtb",
        "mr": "mr_ufal",
        "ur": "ur_udtb",
        "ta": "ta_ttb",
        "te": "te_mtg",
        "th": None,
        "ru": "ru_syntagrus",
        "bg": "bg_btb",
        "he": "he_htb",
        "ka": None,
        "vi": "vi_vtb",
        "fr": "fr_ftb",
        "de": "de_hdt",
    }

    def __init__(self, language, tokenizer, max_length=128, lang_offset=0):
        self.tokenizer = tokenizer
        self.max_length = max_length - 2
        self.language = language
        self.lang_offset = lang_offset

        config = self.lang_to_config[language]
        if config is None:
            raise ValueError("Language not supported")
        self.dataset = load_dataset("universal_dependencies", config)

    def get_sentences(self, examples):
        return examples["tokens"]

    def get_heads(self, examples):
        """Get the head of each token"""
        return examples["head"]

    def get_heads_matrix(self, examples):
        pass

    def tokenize(self, examples) -> (dict, dict):
        inputs = self.tokenizer.batch_encode_plus(
            self.get_sentences(examples),
            truncation=True,
            is_split_into_words=True,
            padding=True,
            max_length=self.max_length,
        )
        # add offset if set for the language
        if self.lang_offset > 0:
            inputs_src["input_ids"] = [
                [tok_id + self.lang_offset if tok_id > 4 else tok_id for tok_id in ii]
                for ii in inputs_src["input_ids"]
            ]

        return inputs

    @property
    def test(self) -> Dataset:
        # return self.dataset.map(self.tokenize, batched=True)
        raise NotImplementedError

    @property
    def train(self):
        raise NotImplementedError

    @property
    def validation(self):
        raise NotImplementedError
