import torch
import json
import argparse
from transformers import set_seed
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
from transformers import DataCollatorWithPadding
from transformers import (
    TrainingArguments,
    Trainer,
    IntervalStrategy,
    EarlyStoppingCallback,
    default_data_collator,
    EvalPrediction,
)
import evaluate
import sys
import os, pickle

from ud_dataset import UDDataset
from utils import load_config, get_tokenizer

from typing import Optional, Union, Tuple
from transformers import (
    XLMRobertaPreTrainedModel,
    XLMRobertaModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
from torch import nn

import numpy as np

import logging

logging.basicConfig(level=logging.INFO)


class XLMRobertaForUD(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)

        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

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
        src: Optional[torch.LongTensor] = None,
        dst: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.probe:
            with torch.no_grad():
                self.roberta.eval()
                outputs = self.roberta(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        else:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        sequence_output = outputs[0]

        # get the contextualized embeddings of the source and destination tokens
        batch_size = sequence_output.shape[0]
        srcs = sequence_output[torch.arange(batch_size), src]
        dsts = sequence_output[torch.arange(batch_size), dst]
        classifier_input = torch.cat([srcs, dsts, srcs * dsts], dim=1)

        logits = self.classifier(classifier_input)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_and_finetune(
    pretrain_input_path,
    ft_output_path,
    model_config,
    max_train_samples,
    max_eval_samples,
    max_test_samples,
    load_checkpoint,
    language,
    seed,
    eval_and_save_steps,
    probe,
    do_train,
    do_eval,
    do_predict,
):
    set_seed(seed)

    logging.info("Loading tokenizer...")
    # get tokenizer:
    tokenizer, lang_offset = get_tokenizer(model_config, language)
    MASK_ID = tokenizer.mask_token_id
    logging.info("Mask token id: " + str(MASK_ID))

    logging.info("Loading dataset...")
    # get dataset
    dataset = UDDataset(
        language,
        tokenizer,
        max_length=model_config["max_sent_len"],
        lang_offset=lang_offset,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        max_test_samples=max_test_samples,
    )
    data_collator = DataCollatorWithPadding(tokenizer)  # for fp16: pad_to_multiple_of=8

    # init trainer:
    logging.info("Loading pretrained model...")
    if not os.path.exists(os.path.join(pretrain_input_path, "config.json")):
        logging.error(
            f"Pretrained model not found at {pretrain_input_path}, finishing."
        )
        return

    model = XLMRobertaForUD.from_pretrained(
        pretrain_input_path, num_labels=dataset.NUM_LABELS
    )
    model.probe = probe
    if probe:
        logging.info("Probing scenario: freezing base model.")
        num_epochs = 30
        # turn off dropout and freeze the base model
        model.roberta.eval()
        for param in model.roberta.parameters():
            param.requires_grad = False
    else:
        num_epochs = 3
    logging.info(f"#params:, {model.num_parameters()}")

    logging.info("Loading pretrain data..")

    os.makedirs(ft_output_path, exist_ok=True)

    # Get the metric function
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)
        f1 = f1_metric.compute(
            predictions=preds, references=p.label_ids, average="macro"
        )
        recall = recall_metric.compute(
            predictions=preds, references=p.label_ids, average="macro"
        )
        precision = precision_metric.compute(
            predictions=preds, references=p.label_ids, average="macro"
        )
        return {**accuracy, **f1, **recall, **precision}

    gradient_accumulation_steps = 1
    training_args = TrainingArguments(
        output_dir=ft_output_path,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=64 // gradient_accumulation_steps,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=eval_and_save_steps,
        eval_steps=eval_and_save_steps,
        dataloader_num_workers=4,
        save_total_limit=1,
        report_to=["tensorboard"],
        evaluation_strategy=IntervalStrategy.STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    logging.info(f"Reporting to: {training_args.report_to}")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset.train if do_train else None,
        eval_dataset=dataset.validation if do_eval else None,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=0.0
            )
        ],
    )

    if do_train:
        logging.info("*** Train ***")
        train_result = trainer.train()

        trainer.save_model(ft_output_path)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        logging.info(f"Done finetune. Finetuned model saved in: {ft_output_path} \n")

        with open(sys.argv[0], "r") as model_code, open(
            os.path.join(ft_output_path, "pretrain_source_code.py"), "w"
        ) as source_out:
            code_lines = model_code.readlines()
            source_out.writelines(code_lines)

    # Evaluation
    if do_eval:
        logging.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        with open(
            os.path.join(ft_output_path, "pretrain_eval.pickle"), "wb"
        ) as evalout:
            pickle.dump(metrics, evalout, protocol=pickle.HIGHEST_PROTOCOL)

    # Prediction
    if do_predict:
        logging.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(
            dataset.test, metric_key_prefix="predict"
        )
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # metric_name = "f1"
        for metric_name in ["accuracy", "f1", "recall", "precision"]:
            out_path = os.path.join(
                ft_output_path, metric_name + "_evaluation", language
            )
            stats_path = os.path.join(out_path, f"{metric_name}_all.txt")
            if os.path.exists(stats_path):
                logging.warning(f"stats_path already exist at {stats_path}.")

            # saving the stats:
            result = metrics[f"predict_{metric_name}"]

            os.makedirs(out_path, exist_ok=True)
            with open(stats_path, "w") as eval_out:
                json.dump({f"eval_{metric_name}": result}, eval_out)

    logging.info("Done.")


def finetune(args):
    logging.info("Finetuning")
    model_config = load_config(args.model_config_path)

    load_and_finetune(
        args.pt_input_path,
        args.ft_output_path,
        model_config,
        args.max_train_samples,
        args.max_eval_samples,
        args.max_test_samples,
        args.load_checkpoint,
        args.language,
        args.seed,
        args.eval_and_save_steps,
        args.probe,
        args.do_train,
        args.do_eval,
        args.do_predict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_input_path", type=str, required=True)
    parser.add_argument("--ft_output_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--max_train_samples", type=int, required=False, default=None)
    parser.add_argument("--max_eval_samples", type=int, required=False, default=None)
    parser.add_argument("--max_test_samples", type=int, required=False, default=None)
    parser.add_argument("--load_checkpoint", action=argparse.BooleanOptionalAction)
    parser.add_argument("--probe", action=argparse.BooleanOptionalAction)
    parser.add_argument("--do_train", action=argparse.BooleanOptionalAction)
    parser.add_argument("--do_eval", action=argparse.BooleanOptionalAction)
    parser.add_argument("--do_predict", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, required=False, default=10)
    parser.add_argument("--eval_and_save_steps", type=int, required=False, default=1000)

    args = parser.parse_args()
    logging.info(vars(args))
    finetune(args)
