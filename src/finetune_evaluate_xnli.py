#!/usr/bin/env python
# coding=utf-8

# This script is adapted from the XNLI example script from the huggingface repository
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_xnli.py

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
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import torch
from xnli_utils import XLMRobertaXNLIHead, XLMRobertaForXNLI

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.26.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


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
    keep_in_memory: bool = field(
        default=False,
        metadata={
            "help": "Keep the dataset in memory instead of writing it to a cache file."
        },
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
        default=True,
        metadata={
            "help": (
                "Will use a custom head for XNLI instead of the one from Huggingface"
            )
        },
    )
    precompute_model_outputs: bool = field(
        default=False,
        metadata={
            "help": (
                "Precompute the model outputs for the dataset and cache them. Makes sense only for probing."
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

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
        # Helping function for adding offset if set for the language
        def _add_offset_to_input_ids(input_ids):
            # Only add offset to the non-special tokens
            return torch.where(input_ids > 4, input_ids + lang_offset, input_ids)

        # Precompute the sentence embeddings
        if model_args.precompute_model_outputs:
            # print("Precomputing the model outputs")
            # Precompute the sentence embeddings for the premise and hypothesis
            inputs = {
                "premise_embedding": examples["premise"],
                "hypothesis_embedding": examples["hypothesis"],
            }
            # First tokenize the texts
            for k, v in inputs.items():
                inputs[k] = tokenizer(
                    v,
                    padding=True,
                    max_length=data_args.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
            # Then compute the sentence embeddings
            with torch.no_grad():
                model.eval()
                for k, v in inputs.items():
                    input_ids = v["input_ids"]
                    if lang_offset > 0:
                        input_ids = _add_offset_to_input_ids(input_ids)

                    inputs[k] = model.compute_sentence_embeddings(
                        input_ids=input_ids.to(device),
                        attention_mask=v["attention_mask"].to(device),
                    )
        else:
            # Tokenize the texts
            inputs = tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding=padding,
                max_length=data_args.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )

            if lang_offset > 0:
                inputs["input_ids"] = _add_offset_to_input_ids(inputs["input_ids"])

        return inputs

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = (
                train_dataset.map(lambda ex: {"premise_len": len(ex["premise"])})
                .sort("premise_len")
                .map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    keep_in_memory=data_args.keep_in_memory,
                    batch_size=128,
                    desc="Running tokenizer on train dataset",
                )
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
                keep_in_memory=data_args.keep_in_memory,
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
                keep_in_memory=data_args.keep_in_memory,
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
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=0.0
            )
        ],
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
