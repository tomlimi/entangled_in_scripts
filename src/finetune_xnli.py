#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import logging
import os
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

import datasets
import numpy as np
from datasets import load_dataset

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    XLMRobertaPreTrainedModel,
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.26.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)

import torch
from torch import nn


class XLMRobertaXNLIHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.out_proj = nn.Linear(config.hidden_size * 3, config.num_labels)

    def forward(self, features_sentence_a, features_sentence_b, **kwargs):
        # the format of features is (batch_size, seq_len, hidden_size)
        mean_a = torch.mean(features_sentence_a, dim=1)
        mean_b = torch.mean(features_sentence_b, dim=1)
        features = torch.cat((mean_a, mean_b, mean_a * mean_b), dim=1)
        x = self.out_proj(features)
        return x


# Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForXNLI(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.roberta_cache = {}
        self.classifier = XLMRobertaXNLIHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. A classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # split input_ids into two parts
        # find the first occurence of the separator token
        sep_token_id = 2  # TODO: do not hardcode this
        # we have input_ids shape (batch_size, seq_len) eg (16, 126)
        # for each sequence we find the separator tokens
        is_sep = (input_ids == sep_token_id).type(torch.uint8)
        sep_indices = is_sep.argmax(dim=1)
        # test using native python
        # assert sep_indices.tolist() == [
        #     i.tolist().index(sep_token_id) for i in input_ids
        # ]

        sep_indices += 1  # add one to account for the separator token
        input_ids_a = torch.ones_like(input_ids)
        attention_mask_a = torch.zeros_like(attention_mask)
        input_ids_b = torch.ones_like(input_ids)
        attention_mask_b = torch.zeros_like(attention_mask)
        for i, j in enumerate(sep_indices):
            # extract the first part of the sequence batch
            input_ids_a[i, :j] = input_ids[i, :j]
            attention_mask_a[i, :j] = 1
            # extract the second part of the sequence batch
            input_ids_b[i, : input_ids.shape[1] - j] = input_ids[i, j:]
            input_ids_b[i, 0] = 0  # TODO: do not hardcode this
            attention_mask_b[i, : input_ids.shape[1] - j] = attention_mask[i, j:]
            # torch.set_printoptions(threshold=10000)
            # print(f"input_ids[{i}]\n", input_ids[i])
            # print(f"attention_mask[{i}]\n", attention_mask[i])
            # print(f"input_ids_a[{i}]\n", input_ids_a[i])
            # print(f"input_ids_b[{i}]\n", input_ids_b[i])
            # print(f"attention_mask_a[{i}]\n", attention_mask_a[i])
            # print(f"attention_mask_b[{i}]\n", attention_mask_b[i])
            # print(f"input_ids_b.shape\n", input_ids_b.shape)
            # print(f"attention_mask_b.shape\n", attention_mask_b.shape)

        assert token_type_ids is None
        assert position_ids is None
        assert head_mask is None
        assert inputs_embeds is None
        assert output_attentions is None
        assert output_hidden_states is None

        outputs_a = self.roberta(
            input_ids_a,
            attention_mask=attention_mask_a,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outputs_b = self.roberta(
            input_ids_b,
            attention_mask=attention_mask_a,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.classifier(outputs_a[0], outputs_b[0])
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        output = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
        return output


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    language: str = field(
        default=None,
        metadata={
            "help": "Evaluation language. Also train language if `train_language` is set to None."
        },
    )
    model_config_path: str = field(
        default=None,
        metadata={"help": "Path to custom model and tokenizer config files."},
    )
    use_custom_xnli_head: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use a custom head for XNLI instead of the one from Huggingface"
            )
        },
    )
    train_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "Train language if it is different from the evaluation language."
        },
    )
    probe: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Train a simple probe on top of the language model with the base model weights frozen."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={
            "help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    if training_args.do_train:
        if model_args.train_language is None:
            train_dataset = load_dataset(
                "xnli",
                model_args.language,
                split="train",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            train_dataset = load_dataset(
                "xnli",
                model_args.train_language,
                split="train",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        label_list = train_dataset.features["label"].names

    if training_args.do_eval:
        eval_dataset = load_dataset(
            "xnli",
            model_args.language,
            split="validation",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        label_list = eval_dataset.features["label"].names

    if training_args.do_predict:
        predict_dataset = load_dataset(
            "xnli",
            model_args.language,
            split="test",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        label_list = predict_dataset.features["label"].names

    # Labels
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="xnli",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Here we load the tokenizer from path specified in model_config
    # if there are more tokenizers, we handle the no-overlap case
    with open(model_args.model_config_path, "r") as fp:
        model_config = json.load(fp)
    if "tokenizer_lang" in model_config:
        # this is the case where we have multiple tokenizers
        language_list = model_config["tokenizer_lang"]
        lang_index = language_list.index(model_args.language)
        tokenizer_path = model_config["tokenizer_path"][lang_index]
    else:
        lang_index = 0
        tokenizer_path = model_config["tokenizer_path"]

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        tokenizer_path,
        max_length=model_config["max_sent_len"],
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # compute the lang_offset. Will be 0 for single tokenizer
    lang_offset = len(tokenizer) * lang_index

    if model_args.use_custom_xnli_head:
        ModelForSequenceClassification = XLMRobertaForXNLI
    else:
        ModelForSequenceClassification = AutoModelForSequenceClassification

    model = ModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    if model_args.probe:
        logging.info("Probing scenario: freezing the base model.")
        for param in model.base_model.parameters():
            param.requires_grad = False

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        inputs = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

        # Add offset if set for the language
        if lang_offset > 0:
            print("Adding offset to the input ids", lang_offset)
            # Only add offset to the non-special tokens
            inputs["input_ids"] = [
                [tok_id + lang_offset if tok_id > 4 else tok_id for tok_id in ii]
                for ii in inputs["input_ids"]
            ]

        return inputs

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Get the metric function
    metric = evaluate.load("xnli")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        )

        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        # trainer.save_metrics("predict", metrics)

        # TODO: move metric_name to arguments
        metric_name = "accuracy"
        out_path = os.path.join(
            training_args.output_dir, metric_name + "_evaluation", model_args.language
        )
        stats = os.path.join(out_path, f"{metric_name}_all.txt")
        if os.path.exists(stats):
            logging.warning(f"Stats already exist at {stats}.")

        # saving the stats:
        result = metrics[f"predict_{metric_name}"]

        os.makedirs(out_path, exist_ok=True)
        with open(os.path.join(out_path, f"{metric_name}_all.txt"), "w") as eval_out:
            json.dump({f"eval_{metric_name}": result}, eval_out)


if __name__ == "__main__":
    main()
