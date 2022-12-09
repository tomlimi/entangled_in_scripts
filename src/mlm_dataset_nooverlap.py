from tqdm import tqdm
import numpy as np
from transformers import set_seed
import os
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
import logging
rng = np.random.RandomState(2021)


class LineByLineTextDataset(Dataset):
    def __init__(self, lang_to_tokenizer, file_paths, block_size, truncate_at=-1, name="", randomize=True, rand_seed=10, is_eval=False, language_codes=None, lang_to_offset=None):
        rng.seed(rand_seed)
        logging.info(f"seed: {rand_seed}")
        lang_ids = []
        input_ids = []

        portion = truncate_at//len(file_paths) if truncate_at!=-1 else -1
        if language_codes:
            logging.info(f'ids: {language_codes}')
        for idx, file_path in enumerate(file_paths):
            new_lines=[]
            logging.info(file_path)
            assert os.path.isfile(file_path), "Input file path {} not found".format(file_path)
            with open(file_path, encoding="utf-8") as f:
                new_lang_lines = [line for line in tqdm(f.readlines(), desc=f"reading lines {name}, is random: {randomize}") if (len(line) > 0 and not line.isspace())]
                new_lines+=new_lang_lines
            if portion>=0 and is_eval:
                new_lines = new_lines[:min(portion,len(new_lines))]

            new_input_ids = lang_to_tokenizer[language_codes[idx]](new_lines, add_special_tokens=True, truncation=True, max_length=block_size-2)['input_ids']
            new_input_ids = [[tok_id + lang_to_offset.get(language_codes[idx], 0) if tok_id > 4 else tok_id for tok_id in ii] for ii in new_input_ids]
            input_ids += new_input_ids

            if language_codes is not None:
                lang_ids += [language_codes[idx]]*len(new_lines)

        if language_codes is not None:
            assert len(input_ids) == len(lang_ids)

        lang_ids = np.array(lang_ids)
        input_ids = np.array(input_ids)

        indices = np.arange(len(input_ids))
        if randomize:
            rng.shuffle(indices)

        if truncate_at >= 1:
            indices=indices[:truncate_at]

        input_ids = input_ids[indices].tolist()

        if language_codes is not None:
            lang_ids = lang_ids[indices].tolist()
        self.examples = input_ids
        if language_codes is not None:
            self.examples = [{"input_ids": torch.tensor(self.examples[i], dtype=torch.long), "language_ids":lang_ids[i]} for i in tqdm(range(len(indices)), desc=f"extracting tokenized lines {name}, order:{indices[:10]}... with ids")]
        else:
            self.examples = [{"input_ids": torch.tensor(self.examples[i], dtype=torch.long), "language_ids":-1} for i in tqdm(range(len(indices)), desc=f"extracting tokenized lines {name}, order:{indices[:10]}...")]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def _collate_batch(examples, tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def tolist(x: Union[List[Any], torch.Tensor]):
    return x.tolist() if isinstance(x, torch.Tensor) else x


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    #lang_to_tokenizer: dict[PreTrainedTokenizerBase]
    # TODO: think how to make special tokens language dependent
    tokenizer: PreTrainedTokenizerBase
    # tokenizer.mask_token_id: int
    # tokenizer.pad_token_id: int
    # tokenizer.cls_token_id: int
    # tokenizer.sep_token_id: int
    vocab_size: int
    mlm: bool = True
    mlm_probability: float = 0.15

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token_id is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # TODO: think how to handle padding when the tokenizers differ for different languages ?
        
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                list(map(lambda x: 1 if x in [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, self.tokenizer.pad_token_id] else 0, val)) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = 5 + torch.randint(self.vocab_size - 5, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    # TODO: Mapping input tokens function based on language indices

