#!/bin/bash
export LD_LIBRARY_PATH=/ha/home/limisiewicz/.virtualenvs/eis/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
cd /home/limisiewicz/my-luster/entangled-in-scripts/entangled_in_scripts || exit 1;


alpha=$1
train_alpha=$2
vocab_size=$3
tok_type=$4
seed=$5
probe=$6
root_path=$7

langs=("ar" "el" "en" "es" "tr" "zh" "hi" "mr" "ur" "ta" "te" "th" "ru" "bg" "he" "vi" "fr" "de")

jids=''
for src_lang in ${langs[@]}
do
  jid=$(sbatch scripts/finetune_pos.sh $alpha $train_alpha $vocab_size $src_lang $tok_type $seed $probe $root_path)
  echo ${jid##* }
  echo $src_lang
  echo $seed
  echo $alpha
  echo $train_alpha
  echo $src_lang

  jids="${jids}:${jid##* }"
  for tgt_lang in ${langs[@]}
  do
    jid_eval=$(sbatch --dependency=afterany:${jid##* } scripts/eval_pos.sh $alpha $train_alpha $vocab_size $src_lang $tgt_lang $tok_type $seed $probe $root_path)
    jids="${jids}:${jid_eval##* }"
  done
done

echo $jids
sbatch --dependency=afterany${jids} scripts/clean_ft_files.sh $alpha $train_alpha $vocab_size $tok_type POS $seed $probe

# source run_pos_pipeline.sh 0.25 0.25 120000 sp-unigram 1234 ..
