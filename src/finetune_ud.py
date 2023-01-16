import torch
import json
import argparse
from transformers import set_seed
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import (
    TrainingArguments,
    Trainer,
    IntervalStrategy,
    EarlyStoppingCallback,
    default_data_collator,
)
import logging
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


class XLMRobertaArcPredictionHead(nn.Module):
    """Head for predicting the head of each input word."""

    def __init__(self, config):
        super().__init__()
        self.out_proj = nn.Linear(config.hidden_size * 3, config.num_labels)

    def forward(self, mean_a, mean_b, **kwargs):
        features = torch.cat((mean_a, mean_b, mean_a * mean_b), dim=1)
        x = self.out_proj(features)
        return x


# Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForSequenceClassification(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = XLMRobertaClassificationHead(config)

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
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
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


# Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForUD(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = XLMRobertaArcPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def compute_contextualized_embeddings(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # the format of features is (batch_size, seq_len, hidden_size)
        return outputs.last_hidden_state

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
            config.num_labels - 1]`. A classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if premise_embedding is None or hypothesis_embedding is None:
            assert input_ids is not None

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

            premise_embedding = self.compute_sentence_embeddings(
                input_ids_a, attention_mask_a
            )
            hypothesis_embedding = self.compute_sentence_embeddings(
                input_ids_b, attention_mask_b
            )

        logits = self.classifier(premise_embedding, hypothesis_embedding)
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


def load_and_finetune(
    pretrain_input_path,
    ft_output_path,
    model_config,
    truncate_at,
    load_checkpoint,
    language,
    seed,
    eval_and_save_steps,
    probe,
):
    set_seed(seed)
    # set verbosity
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logging.info("Loading tokenizer...")
    # get tokenizer:
    tokenizer, lang_offset = get_tokenizer(model_config)
    MASK_ID = tokenizer.mask_token_id
    logging.info("Mask token id: " + str(MASK_ID))

    logging.info("Loading dataset...")
    # get dataset
    dataset = UDDataset(
        language,
        tokenizer,
        truncate_at=truncate_at,
        max_length=model_config["max_sent_len"],
        lang_offset=lang_offset,
    )
    data_collator = default_data_collator

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
    gradient_accumulation_steps = 1

    training_args = TrainingArguments(
        output_dir=ft_output_path,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16 // gradient_accumulation_steps,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=eval_and_save_steps,
        eval_steps=eval_and_save_steps,
        save_total_limit=2,
        report_to=["tensorboard"],
        evaluation_strategy=IntervalStrategy.STEPS,
        load_best_model_at_end=True,
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    logging.info(f"Reporting to: {training_args.report_to}")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset.train,
        eval_dataset=dataset.validation,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=0.0
            )
        ],
    )

    if load_checkpoint:
        logging.info("loading finetuning checkpoint")
        try:
            trainer.train(resume_from_checkpoint=True)
        except Exception as e:
            logging.info("Failed loading checkpoint, regular training")
            trainer.train()
    else:
        logging.info("Finetuning from scratch")
        trainer.train()

    trainer.save_model(ft_output_path)
    logging.info(f"Done finetune. Finetuned model saved in: {ft_output_path} \n")
    metrics = trainer.evaluate()
    logging.info(metrics)

    with open(os.path.join(ft_output_path, "pretrain_eval.pickle"), "wb") as evalout:
        pickle.dump(metrics, evalout, protocol=pickle.HIGHEST_PROTOCOL)

    with open(sys.argv[0], "r") as model_code, open(
        os.path.join(ft_output_path, "pretrain_source_code.py"), "w"
    ) as source_out:
        code_lines = model_code.readlines()
        source_out.writelines(code_lines)

    logging.info("Done.")


def finetune(args):
    logging.info("Finetuning")
    model_config = load_config(args.model_config_path)

    load_and_finetune(
        args.pt_input_path,
        args.ft_output_path,
        model_config,
        args.truncate_at,
        args.load_checkpoint,
        args.language,
        args.seed,
        args.eval_and_save_steps,
        args.probe,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_input_path", type=str, required=True)
    parser.add_argument("--ft_output_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--truncate_at", type=int, required=False, default=-1)
    parser.add_argument("--load_checkpoint", action=argparse.BooleanOptionalAction)
    parser.add_argument("--probe", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, required=False, default=10)
    parser.add_argument("--eval_and_save_steps", type=int, required=False, default=1000)

    args = parser.parse_args()
    logging.info(vars(args))
    finetune(args)
