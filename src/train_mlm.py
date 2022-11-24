import torch
import argparse
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from transformers import set_seed
from transformers import XLMRobertaTokenizerFast
from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM


import logging
import sys
import os, pickle
import json
import numpy as np
from mlm_dataset import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, IntervalStrategy
from eval import compute_metrics

logging.basicConfig(level=logging.INFO)
logging.info(torch.cuda.is_available())


def pretrain(pretrain_outpath, model_config, pt_config, truncate_at, load_checkpoint=True, data_seed=10, seed=10, eval_and_save_steps=5000, early_stopping_patience=2, initial_learning_rate=5e-5, gradient_accumulation_steps=8, fp16=True, gradient_checkpointing=False):
    set_seed(seed)

    logging.info("Loading tokenizer..")
    # get tokenizer:
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_config['tokenizer_path'], max_len=model_config['max_sent_len'])
    MASK_ID = tokenizer.mask_token_id
    logging.info(MASK_ID)
    # build dataset:
    logging.info("Building Model..")
    
    config = XLMRobertaConfig(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_layer_size'],
        num_hidden_layers=model_config['num_hidden'],
        num_attention_heads=model_config['num_attention'],
        max_position_embeddings=model_config['max_sent_len']
    )
    model = XLMRobertaForMaskedLM(config)
    # define a data_collator (a small helper that will help us batch different samples of the dataset together into an
    # object that PyTorch knows how to perform backprop on):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    # init trainer:
    logging.info("Training\Loading model...")
    logging.info(f"#params:, {model.num_parameters()}")
    if torch.cuda.is_available():
        logging.info(f"#memory used:, {memory_used_in_mb()} MB")

    if not os.path.exists(os.path.join(pretrain_outpath,'config.json')):
        logging.info("Loading pretrain data..")
        pretrain_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_paths= pt_config['train_data_paths_list'], block_size=model_config['max_sent_len'], truncate_at=truncate_at, name="pretrain train", rand_seed=data_seed)
        preeval_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_paths=pt_config['eval_data_paths_list'], block_size=model_config['max_sent_len'], truncate_at=truncate_at, name="pretrain eval", rand_seed=data_seed, is_eval=True)
        logging.info("Pretraining model..")
        os.makedirs(pretrain_outpath, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=pretrain_outpath,
            overwrite_output_dir=True,
            num_train_epochs=pt_config['num_epochs'],
            per_device_train_batch_size=pt_config['batch_size']//gradient_accumulation_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=(fp16 and torch.cuda.is_available()),
            gradient_checkpointing=gradient_checkpointing,
            save_steps=eval_and_save_steps,
            per_device_eval_batch_size=32,
            save_total_limit=5,
            report_to=['tensorboard'],
            eval_steps=eval_and_save_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_accumulation_steps=1,
            metric_for_best_model='mrr',
            greater_is_better=True,
            load_best_model_at_end=True,
            learning_rate=initial_learning_rate
        )
        logging.info(f"Reporting to: {training_args.report_to}")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=pretrain_dataset,
            eval_dataset=preeval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=0.0)]
        )
        if load_checkpoint:
            logging.info("loading pt checkpoint")
            try:
                trainer.train(resume_from_checkpoint=True)
            except Exception as e:
                logging.info("Failed loading checkpoint, regular training")
                trainer.train()
        else:
            logging.info("training pt from scratch")
            trainer.train()
        trainer.save_model(pretrain_outpath)
        logging.info(f"Done pretrain. pretrained model saved in: {pretrain_outpath} \n")
        metrics = trainer.evaluate()
        logging.info(metrics)

        with open(os.path.join(pretrain_outpath,'pretrain_eval.pickle'), 'wb') as evalout:
            pickle.dump(metrics, evalout, protocol=pickle.HIGHEST_PROTOCOL)

        with open(sys.argv[0], 'r') as model_code, open(os.path.join(pretrain_outpath,'pretrain_source_code.py'), 'w') as source_out :
            code_lines = model_code.readlines()
            source_out.writelines(code_lines)

        logging.info("Only pretrain, Done.")
    else:
        logging.info(f"model exists: {pretrain_outpath}")


def load_config(config_path):
    with open(config_path, 'r') as fp:
        return json.load(fp)


def memory_used_in_mb():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2


def train(args):
    out_path = args.out
    pretrain_name = args.pretrain_name
    data_seed = args.data_seed
    seed = args.seed
    logging.info('Training mulitling')

    pretrain_outpath = os.path.join(out_path, pretrain_name+'_'+str(seed))
    logging.info(f"Common pretrain path: {pretrain_outpath}")

    model_config = load_config(args.model_config_path)
    pt_config = load_config(args.pretrain_config_path)

    pretrain(pretrain_outpath, model_config, pt_config, args.truncate_at, args.load_checkpoint, data_seed, seed=seed, eval_and_save_steps=args.eval_and_save_steps, early_stopping_patience=args.early_stopping_patience, initial_learning_rate=args.initial_learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', type=str, required=True)
    parser.add_argument('-p', '--pretrain_name', type=str, required=True)
    parser.add_argument('--model_config_path',type=str, required=True)
    parser.add_argument('--pretrain_config_path',type=str, required=False, default=None)
    parser.add_argument('--truncate_at',type=int, required=False, default=-1)
    parser.add_argument('--load_checkpoint',type=bool, required=False, default=True)
    parser.add_argument('--data_seed',type=int, required=False, default=10)
    parser.add_argument('--seed',type=int, required=False, default=10)
    parser.add_argument('--eval_and_save_steps', type=int, required=False, default=5000)
    parser.add_argument('--early_stopping_patience', type=int, required=False, default=20)
    parser.add_argument('--initial_learning_rate', type=float, required=False, default=2e-2)

    args = parser.parse_args()
    logging.info(vars(args))
    train(args)
