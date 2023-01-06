import torch
import json
import argparse
from transformers import set_seed
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
# from transformers import XLMBertTokenizer, BertForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer, IntervalStrategy, EarlyStoppingCallback
import logging
import sys
import os, pickle

from classification_dataset import XtremePOSClassificationDataset, XtremeNERClassificationDataset
from utils import load_config


def load_and_finetune(pretrain_in_path, ft_out_path, model_config, truncate_at, load_checkpoint, language, task='POS',
                      seed=10, eval_and_save_steps=1000,probe=True):

    set_seed(seed)
    logging.info("Loading tokenizer...")
    # get tokenizer:
    if 'tokenizer_lang' in model_config:
        language_list = model_config['tokenizer_lang']
        lang_index = language_list.index(language)
    
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_config['tokenizer_path'][lang_index], max_length=model_config['max_sent_len'])
        lang_offset = len(tokenizer) * lang_index
    else:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_config['tokenizer_path'], max_length=model_config['max_sent_len'])
        lang_offset = 0

    MASK_ID = tokenizer.mask_token_id
    logging.info(MASK_ID)

    logging.info("Loading dataset...")
    # get dataset
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    if task == 'POS':
        dataset = XtremePOSClassificationDataset(language, tokenizer, truncate_at=truncate_at,
                                                 max_length=model_config['max_sent_len'], lang_offset=lang_offset)
    elif task == 'NER':
        dataset = XtremeNERClassificationDataset(language, tokenizer, truncate_at=truncate_at,
                                                 max_length=model_config['max_sent_len'], lang_offset=lang_offset)
    else:
        raise ValueError(f"Unsupported task: {task}. Only `POS` is currently supported.")

    # init trainer:
    logging.info("Loading pretrained model...")
    if not os.path.exists(os.path.join(pretrain_in_path,'config.json')):
        logging.warning(f"Pretrained model not found at {pretrain_in_path}, finishing.")
        return

    model = XLMRobertaForTokenClassification.from_pretrained(pretrain_in_path, num_labels=dataset.NUM_LABELS)
    if probe:
        logging.info("Probing scenario: freezeing base model.")
        num_epochs = 30
        for param in model.base_model.parameters():
            param.requires_grad = False
    else:
        num_epochs = 3
    logging.info(f"#params:, {model.num_parameters()}")

    logging.info("Loading pretrain data..")

    os.makedirs(ft_out_path, exist_ok=True)
    gradient_accumulation_steps = 4
    
    training_args = TrainingArguments(
        output_dir=ft_out_path,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16//gradient_accumulation_steps,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=eval_and_save_steps,
        eval_steps=eval_and_save_steps,
        save_total_limit=3,
        report_to=['tensorboard'],
        evaluation_strategy=IntervalStrategy.STEPS,
        load_best_model_at_end=True,
        learning_rate=2e-5,
        weight_decay=0.01
    )
    logging.info(f"Reporting to: {training_args.report_to}")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset.train,
        eval_dataset=dataset.validation,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)]
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

    trainer.save_model(ft_out_path)
    logging.info(f"Done finetune. Finetuned model saved in: {ft_out_path} \n")
    metrics = trainer.evaluate()
    logging.info(metrics)

    with open(os.path.join(ft_out_path,'pretrain_eval.pickle'), 'wb') as evalout:
        pickle.dump(metrics, evalout, protocol=pickle.HIGHEST_PROTOCOL)

    with open(sys.argv[0], 'r') as model_code, open(os.path.join(ft_out_path,'pretrain_source_code.py'), 'w') as source_out :
        code_lines = model_code.readlines()
        source_out.writelines(code_lines)

    logging.info("Done.")


def finetune(args):

    logging.info('Finetuning')
    seed_in = args.seed_in
    seed = args.seed
    lang = args.language
    task = args.ft_task
    model_config = load_config(args.model_config_path)

    pt_in_path = os.path.join(args.pretrain_path, args.pretrain_name + '_' +str(seed_in))
    ft_output_path = os.path.join(args.finetune_path, args.pretrain_name+ '_' +str(seed), lang)

    logging.info(f"pt in: {pt_in_path}")
    logging.info(f"ft out: {ft_output_path}")

    load_and_finetune(pt_in_path, ft_output_path, model_config,  args.truncate_at, args.load_checkpoint, lang, task=task,
                      seed=seed, eval_and_save_steps=args.eval_and_save_steps, probe=args.probe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--finetune_path', type=str, required=True)
    parser.add_argument('-i', '--pretrain_path', type=str, required=True)
    parser.add_argument('-p', '--pretrain_name', type=str, required=True)
    parser.add_argument('-l', '--language', type=str, required=True)
    parser.add_argument('--ft_task', type=str, default='POS', required=False)
    parser.add_argument('--model_config_path',type=str, required=True)
    parser.add_argument('--truncate_at',type=int, required=False, default=-1)
    parser.add_argument('--load_checkpoint',type=bool, required=False, default=True)
    parser.add_argument('--probe', type=lambda x: (str(x).lower() == 'true'), required=False, default=True)
    parser.add_argument('--seed_in',type=int, required=False, default=1234)
    parser.add_argument('--seed',type=int, required=False, default=10)
    parser.add_argument('--eval_and_save_steps', type=int, required=False, default=1000)

    args = parser.parse_args()
    logging.info(vars(args))
    finetune(args)
