import torch
import json
import argparse
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
#from transformers import BertTokenizer, BertForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from datasets import load_metric
import logging
import os
import numpy as np

from classification_dataset import XtremePOSClassificationDataset, XtremeNERClassificationDataset


def evaluate_metric(dataset, model, data_collator, metric_name, task='POS'):

    ner_mapping = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}

    if metric_name == 'f1':
        eval_metric = load_metric('seqeval')
    elif metric_name == 'f1-macro':
        eval_metric = load_metric('f1')
    else:
        eval_metric = load_metric(metric_name)

    eval_trainer = Trainer(
        model=model,
        data_collator=data_collator
    )

    predictions, labels, _ = eval_trainer.predict(dataset)
    predictions = np.argmax(predictions, axis=2)
    for prediction, label in zip(predictions, labels):
        true_predictions = [p for (p, l) in zip(prediction, label) if l != -100]
        true_labels = [l for (p, l) in zip(prediction, label) if l != -100]
        if task == 'NER' and metric_name == 'f1':
            true_predictions = [[ner_mapping[p] for p in true_predictions]]
            true_labels = [[ner_mapping[l] for l in true_labels]]
        eval_metric.add_batch(predictions=true_predictions, references=true_labels)

    if metric_name == 'f1':
        eval_results = eval_metric.compute()
        return eval_results['overall_f1']
    elif metric_name == 'f1-macro':
        eval_results = eval_metric.compute(average='macro')
        return eval_results['f1']
    eval_results = eval_metric.compute()
    return eval_results[metric_name]


def eval(args):
    """
    runs a single evaluation for classification performance
    """
    # initializing variables w.r.t args:
    src_lang = args.language_src
    tgt_lang = args.language_tgt
    model_config_path = args.model_config_path
    truncate_at = args.truncate_at
    overrite = args.overrite
    seed = args.seed
    metric_name = args.metric
    task = args.ft_task

    ft_in_path = os.path.join(args.finetune_path, args.pretrain_name + '_' + str(seed), src_lang)
    out_path = os.path.join(args.finetune_path, args.pretrain_name + '_' + str(seed), src_lang, metric_name + '_evaluation', tgt_lang)
    model_config = json.load(open(model_config_path, 'r'))

    logging.info(f"Output is: {out_path}")
    if os.path.exists(os.path.join(out_path,f'{metric_name}_all.txt')) and not overrite:
        logging.info(f"stats already exist at {os.path.join(out_path,f'{metric_name}_all.txt')}, no overrite. finishing.")
        return

    logging.info(f"Loading tokenizer")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_config['tokenizer_path'])

    logging.info("Loading dataset...")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    if task == 'POS':
        dataset = XtremePOSClassificationDataset(tgt_lang, tokenizer,
                                                 truncate_at=truncate_at, max_length=model_config['max_sent_len'])
    elif task == 'NER':
        dataset = XtremeNERClassificationDataset(tgt_lang, tokenizer,
                                                 truncate_at=truncate_at, max_length=model_config['max_sent_len'])
    else:
        raise ValueError(f"Unaupported task: {task}. Only `POS` is currently supported.")

    logging.info(f"Loading model")
    model = XLMRobertaForTokenClassification.from_pretrained(ft_in_path)
    logging.info(f"#params:, {model.num_parameters()}")

    logging.info("Gathering stats...")
    logging.info(f"Evaulating {task} in {tgt_lang} finetuned: {src_lang} with truncate {truncate_at}. Overrite:{overrite}.")
    # gathering scores:

    result = evaluate_metric(dataset.test, model, data_collator, metric_name, task=task)

    logging.info(f"BERT {metric_name}: {result}")
    # saving the stats:

    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path,f'{metric_name}_all.txt'), 'w') as eval_out:
        json.dump({f'eval_{metric_name}': result}, eval_out)
    with open(os.path.join(out_path,f'script_args.txt'), 'w') as script_args:
        json.dump({'args':str(vars(args)),'note':'overrite is manually always True.'}, script_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--finetune_path', type=str, required=True)
    parser.add_argument('-p', '--pretrain_name', type=str, required=True)
    parser.add_argument('-s', '--language_src', type=str, required=True)
    parser.add_argument('-t', '--language_tgt', type=str, required=True)
    parser.add_argument('--ft_task', type=str, default='POS', required=False)
    parser.add_argument('--metric', type=str, default='accuracy', required=False)
    parser.add_argument('--overrite', type=bool, default=True)
    parser.add_argument('--model_config_path',type=str, required=True)
    parser.add_argument('--truncate_at',type=int, required=False, default=-1)
    parser.add_argument('--load_checkpoint',type=bool, required=False, default=True)
    parser.add_argument('--seed',type=int, required=False, default=1234)
    args = parser.parse_args()
    logging.info(vars(args))
    eval(args)
