#!/bin/bash

cd /home/limisiewicz/my-luster/entangled-in-scripts/entangled_in_scripts || exit 1;


alpha=$1
train_alpha=$2
vocab_size=$3
seed=$4

for src_lang in 'ar' 'el' 'en' 'es' 'tr' 'zh'
do
  # jid=$(sbatch --killable --requeue scripts/finetune_ner.sh $alpha $train_alpha $vocab_size $src_lang $seed)
  jid=$(sbatch scripts/finetune_ner.sh $alpha $train_alpha $vocab_size $src_lang $seed)
  echo ${jid##* }
  echo $src_lang
  echo $seed
  echo $alpha
  echo $train_alpha
  echo $src_lang

  for tgt_lang in 'ar' 'el' 'en' 'es' 'tr' 'zh'
  do
    # sbatch --killable --requeue --dependency=afterany:${jid##* } scripts/eval_ner.sh $alpha $train_alpha $vocab_size $src_lang $tgt_lang $seed
    # sbatch --killable --requeue scripts/eval_ner.sh $alpha $train_alpha $vocab_size $src_lang $tgt_lang $seed
    sbatch --dependency=afterany:${jid##* } scripts/eval_ner.sh $alpha $train_alpha $vocab_size $src_lang $tgt_lang $seed
  done
done

# source run_ner_pipeline.sh 0.25 0.25 120000 1234

