from typing import Optional, Union, Tuple
from transformers import (
    XLMRobertaPreTrainedModel,
    XLMRobertaModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
from torch import nn


class XLMRobertaXNLIHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.out_proj = nn.Linear(config.hidden_size * 3, config.num_labels)

    def forward(self, mean_a, mean_b, **kwargs):
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
        self.classifier = XLMRobertaXNLIHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def compute_sentence_embeddings(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # the format of features is (batch_size, seq_len, hidden_size)
        masked_features = outputs[0] * attention_mask.unsqueeze(-1)
        masked_mean = torch.sum(masked_features, dim=1) / torch.sum(
            attention_mask, dim=1, keepdim=True
        )

        return masked_mean

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
        premise_embedding: Optional[torch.FloatTensor] = None,
        hypothesis_embedding: Optional[torch.FloatTensor] = None,
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
