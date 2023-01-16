import torch
from abc import abstractmethod
from datasets import Dataset, load_dataset
import transformers
import random
import logging

logging.basicConfig(level=logging.INFO)


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

    def __init__(
        self,
        language,
        tokenizer,
        max_length,
        lang_offset,
        max_train_samples,
        max_eval_samples,
        max_test_samples,
    ):
        self.language = language
        self.tokenizer = tokenizer
        self.max_length = max_length - 2
        self.lang_offset = lang_offset
        self.padding = False
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.max_test_samples = max_test_samples

        config = self.lang_to_config[language]
        if config is None:
            raise ValueError("Language not supported")
        self.dataset = load_dataset("universal_dependencies", config)

    @staticmethod
    def tokenize(examples, tokenizer, lang_offset, padding, max_length):
        inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=padding,
            max_length=max_length,
        )
        # add offset if set for the language
        if lang_offset > 0:
            inputs["input_ids"] = [
                [tok_id + lang_offset if tok_id > 4 else tok_id for tok_id in ii]
                for ii in inputs["input_ids"]
            ]

        return inputs

    @staticmethod
    def align_labels(examples, tokenizer, max_length):
        examples_labels = []
        for text, words, heads, input_ids in zip(
            examples["text"],
            examples["tokens"],
            examples["head"],
            examples["input_ids"],
        ):
            assert len(words) == len(heads)
            # new_input_ids = [0]
            labels = []
            labels.append(-100)
            token_index_mapping = [0]
            for i, (word, head) in enumerate(zip(words, heads)):
                subword_tokens = tokenizer.encode(word, add_special_tokens=False)
                token_index_mapping.append(len(labels))
                # new_input_ids.extend(subword_tokens)
                first_subword = True
                for _ in subword_tokens:
                    if first_subword:
                        labels.append(int(head) if head != "None" else -100)
                        first_subword = False
                    else:
                        labels.append(-100)
            # renumber labels to the first subwords
            for i in range(len(labels)):
                if labels[i] != -100:
                    labels[i] = token_index_mapping[labels[i]]
                    # skip labels that are out of range
                    if labels[i] >= max_length - 1:
                        labels[i] = -100
            labels = labels[: max_length - 1]  # truncating labels
            labels.append(-100)
            # pad labels
            # while len(labels) < self.max_length - 1:
            #     labels.append(-100)

            # new_input_ids = new_input_ids[: self.max_length - 1]  # truncating input_ids
            # new_input_ids.append(2)
            # print(new_input_ids, input_ids)
            # assert new_input_ids == input_ids

            for label in labels:
                assert label >= -100 and label < len(input_ids)
            examples_labels.append(labels)

        return {"labels": examples_labels}

    @staticmethod
    def generate_arc_prediction_examples(examples, subsample_negative):
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
            valid_heads = [i for i in range(len(labels)) if labels[i] != -100]

            for src in range(1, len(labels) - 1):  # skip root and eos
                # skip padding / inner-tokens
                if labels[src] == -100:
                    continue
                for dst in range(len(labels) - 1):  # skip eos
                    # skip padding / inner-tokens
                    if labels[dst] == -100:
                        continue

                    label = int(labels[src] == dst)
                    # balance positive and negative examples
                    if label == 0 and subsample_negative:
                        if random.random() > 1 / (len(valid_heads)):
                            continue
                    if src != dst:
                        new_examples["input_ids"].append(input_ids)
                        new_examples["attention_mask"].append(attention_mask)
                        new_examples["labels"].append(label)
                        new_examples["src"].append(src)
                        new_examples["dst"].append(dst)
        logging.info(
            "ratio of positive examples: %f",
            sum(new_examples["labels"]) / len(new_examples["labels"]),
        )
        return new_examples

    def _prepare_dataset(self, dataset, truncate_at, subsample_negative):
        logging.info("Tokenizing dataset...")
        dataset = dataset.map(
            lambda x: UDDataset.tokenize(
                x, self.tokenizer, self.lang_offset, self.padding, self.max_length
            ),
            batched=True,
        )
        logging.info("Aligning labels...")
        dataset = dataset.map(
            lambda x: UDDataset.align_labels(x, self.tokenizer, self.max_length),
            batched=True,
        )
        logging.info("Generating arc prediction examples...")
        dataset = dataset.map(
            lambda x: UDDataset.generate_arc_prediction_examples(x, subsample_negative),
            batched=True,
            remove_columns=dataset.column_names,
        )
        if truncate_at is not None:
            logging.info("Truncating dataset...")
            dataset = dataset.shuffle(42).select(range(truncate_at))
        # dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    @property
    def train(self):
        return self._prepare_dataset(
            self.dataset["train"], self.max_train_samples, subsample_negative=True
        )

    @property
    def validation(self):
        return self._prepare_dataset(
            self.dataset["validation"], self.max_eval_samples, subsample_negative=False
        )

    @property
    def test(self) -> Dataset:
        return self._prepare_dataset(
            self.dataset["test"], self.max_test_samples, subsample_negative=False
        )
