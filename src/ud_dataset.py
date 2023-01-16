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
    NUM_LABELS = 2

    def __init__(self, language, tokenizer, truncate_at, max_length, lang_offset):
        self.language = language
        self.tokenizer = tokenizer
        self.truncate_at = truncate_at
        self.max_length = max_length - 2
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

    def tokenize(self, examples):
        inputs = self.tokenizer.batch_encode_plus(
            self.get_sentences(examples),
            truncation=True,
            is_split_into_words=True,
            padding=False,
            max_length=self.max_length,
        )
        # add offset if set for the language
        if self.lang_offset > 0:
            inputs["input_ids"] = [
                [tok_id + self.lang_offset if tok_id > 4 else tok_id for tok_id in ii]
                for ii in inputs["input_ids"]
            ]

        return inputs

    def align_labels(self, examples):
        examples_labels = []
        for text, words, heads, input_ids in zip(
            examples["text"],
            examples["tokens"],
            examples["head"],
            examples["input_ids"],
        ):
            assert len(words) == len(heads)
            new_input_ids = [0]
            labels = []
            labels.append(-100)
            token_index_mapping = [0]
            for i, (word, head) in enumerate(zip(words, heads)):
                subword_tokens = self.tokenizer.encode(word, add_special_tokens=False)
                token_index_mapping.append(len(new_input_ids))
                new_input_ids.extend(subword_tokens)
                first_subword = True
                for _ in subword_tokens:
                    if first_subword:
                        labels.append(int(head) if head != "None" else -100)
                        first_subword = False
                    else:
                        labels.append(-100)
            for i in range(len(labels)):
                if labels[i] != -100:
                    labels[i] = token_index_mapping[labels[i]]
            labels = labels[: self.max_length - 1]  # truncating labels
            new_input_ids = new_input_ids[: self.max_length - 1]  # truncating input_ids
            new_input_ids.append(2)
            labels.append(-100)
            # print(token_index_mapping)
            # print(text, words, heads)
            # print(new_input_ids)
            print(input_ids)
            print(labels)
            print()
            assert new_input_ids == input_ids
            examples_labels.append(labels)

        return {"labels": examples_labels}

    def generate_arc_prediction_examples(self, examples):
        new_examples = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "src": [],
            "dst": [],
        }
        for input_ids, attention_mask, labels in zip(
            examples["input_ids"], examples["attention_mask"], examples["labels"]
        ):
            for src in range(1, len(labels) - 1):  # skip root and eos
                # also skip subword tokens that are not the first subword
                if labels[src] == -100:
                    continue
                for dst in range(len(labels) - 1):  # skip eos
                    if src != dst:
                        new_examples["input_ids"].append(input_ids)
                        new_examples["attention_mask"].append(attention_mask)
                        new_examples["labels"].append(int(labels[src] == dst))
                        new_examples["src"].append(src)
                        new_examples["dst"].append(dst)
        return new_examples

    def _prepare_dataset(self, dataset):
        dataset = dataset.map(lambda x: self.tokenize(x), batched=True)
        dataset = dataset.map(lambda x: self.align_labels(x), batched=True)
        dataset = dataset.map(
            self.generate_arc_prediction_examples,
            batched=True,
            remove_columns=dataset.column_names,
        )
        if self.truncate_at is not None:
            dataset = dataset.shuffle().select(range(self.truncate_at))
        breakpoint()
        # dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    @property
    def test(self) -> Dataset:
        return self._prepare_dataset(self.dataset["test"])

    @property
    def train(self):
        return self._prepare_dataset(self.dataset["train"])

    @property
    def validation(self):
        return self._prepare_dataset(self.dataset["validation"])
