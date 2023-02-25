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
        "ru": "ru_syntagrus",
        "bg": "bg_btb",
        "he": "he_htb",
        "vi": "vi_vtb",
        "fr": "fr_ftb",
        "de": "de_hdt",
    }
    deprel_id = [
        "punct",
        "case",
        "det",
        "nmod",
        "nsubj",
        "obl",
        "amod",
        "advmod",
        "obj",
        "root",
        "cc",
        "conj",
        "mark",
        "aux",
        "nummod",
        "compound",
        "flat",
        "xcomp",
        "fixed",
        "acl",
        "ccomp",
        "appos",
        "advcl",
        "cop",
        "iobj",
        "parataxis",
        "csubj",
        "dep",
        "expl",
        "discourse",
        "clf",
        "orphan",
        "list",
        "dislocated",
        "vocative",
        "goeswith",
        "reparandum",
    ]
    deprel_id = {k: i for i, k in enumerate(deprel_id)}
    print(deprel_id)

    NUM_LABELS = len(deprel_id)

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

        config = self.lang_to_config.get(language, None)
        # limit the number of training samples for russian and german languages
        if config == "ru_syntagrus" or config == "de_hdt":
            logging.info("Limiting the number of training samples to 15000")
            self.max_train_samples = 15000

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
        all_new_heads = []
        all_new_deprels = []
        for text, words, heads, deprels, input_ids in zip(
            examples["text"],
            examples["tokens"],
            examples["head"],
            examples["deprel"],
            examples["input_ids"],
        ):
            assert len(words) == len(heads)
            assert len(words) == len(deprels)

            # create the mapping from word index to first subword index
            token_index_mapping = [0]  # root token at the beginning
            current_index = 1
            for i, word in enumerate(words):
                subword_tokens = tokenizer.encode(word, add_special_tokens=False)
                token_index_mapping.append(current_index)
                current_index += len(subword_tokens)

            assert len(token_index_mapping) == len(words) + 1  # +1 for the root token

            # create the new labels
            new_heads = [-100] * len(input_ids)
            new_deprels = [-100] * len(input_ids)
            for subword_index, head, deprel in zip(
                token_index_mapping[1:], heads, deprels
            ):
                # skip labels that are out of range (-1 for the end of sentence token)
                if subword_index >= max_length - 1:
                    continue
                # 1. handle heads
                head = int(head) if head != "None" else -100
                #   map the destination of the arc
                if head != -100:
                    head = token_index_mapping[head]
                #   check if the label is in range
                if head < max_length - 1:
                    new_heads[subword_index] = head
                # 2. handle deprels
                if ":" in deprel:
                    deprel = deprel.split(":")[0]
                deprel = (
                    UDDataset.deprel_id[deprel]
                    if deprel in UDDataset.deprel_id
                    else -100
                )
                new_deprels[subword_index] = deprel

            assert len(new_heads) == len(input_ids)
            assert len(new_deprels) == len(input_ids)

            for head in new_heads:
                assert head >= -100 and head < len(input_ids)
            all_new_heads.append(new_heads)
            all_new_deprels.append(new_deprels)

        return {"head": all_new_heads, "deprel": all_new_deprels}

    @staticmethod
    def generate_arc_prediction_examples(examples):
        new_examples = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "src": [],
            "dst": [],
        }
        for input_ids, attention_mask, heads, deprels in zip(
            examples["input_ids"],
            examples["attention_mask"],
            examples["head"],
            examples["deprel"],
        ):
            n = len(input_ids)
            for dst in range(n):
                head = heads[dst]
                deprel = deprels[dst]
                if head == -100 or deprel == -100:
                    continue
                new_examples["input_ids"].append(input_ids)
                new_examples["attention_mask"].append(attention_mask)
                new_examples["labels"].append(deprel)
                new_examples["src"].append(head)
                new_examples["dst"].append(dst)
        return new_examples

    def _prepare_dataset(self, dataset, truncate_at):
        load_from_cache_file = True
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
            load_from_cache_file=load_from_cache_file,
        )
        logging.info("Generating arc prediction examples...")
        dataset = dataset.map(
            lambda x: UDDataset.generate_arc_prediction_examples(x),
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=load_from_cache_file,
        )
        if truncate_at is not None:
            logging.info("Truncating dataset...")
            truncate_at = min(truncate_at, len(dataset))
            dataset = dataset.shuffle(42).select(range(truncate_at))
        # dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    @property
    def train(self):
        train_dataset = self._prepare_dataset(
            self.dataset["train"], self.max_train_samples
        )

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logging.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        return train_dataset

    @property
    def validation(self):
        return self._prepare_dataset(self.dataset["validation"], self.max_eval_samples)

    @property
    def test(self) -> Dataset:
        return self._prepare_dataset(self.dataset["test"], self.max_test_samples)
