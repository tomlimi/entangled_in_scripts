#!/bin/bash
export LD_LIBRARY_PATH=/ha/home/limisiewicz/.virtualenvs/eis/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
cd /home/limisiewicz/my-luster/entangled-in-scripts/entangled_in_scripts || exit 1;


alpha=$1
train_alpha=$2
vocab_size=$3
tok_type=$4
seed=$5
probe=$6
# langs=("ar" "el" "en" "es" "tr" "zh")
langs=("ar" "el" "en" "es" "tr" "zh" "hi" "mr" "ur" "ta" "te" "th" "ru" "bg" "he" "vi" "fr" "de")

for src_lang in ${langs[@]}
do
  #jid=$(sbatch --killable --requeue scripts/finetune_pos.sh $alpha $train_alpha $vocab_size $src_lang $seed)
  jid=$(sbatch scripts/finetune_pos.sh $alpha $train_alpha $vocab_size $src_lang $tok_type $seed $probe)
  echo ${jid##* }
  echo $src_lang
  echo $seed
  echo $alpha
  echo $train_alpha
  echo $src_lang

  for tgt_lang in ${langs[@]}
  do
    # sbatch --killable --requeue --dependency=afterany:${jid##* } scripts/eval_pos.sh $alpha $train_alpha $vocab_size $src_lang $tgt_lang $seed
    # sbatch scripts/eval_pos.sh $alpha $train_alpha $vocab_size $src_lang $tgt_lang $tok_type $seed
    sbatch --dependency=afterany:${jid##* } scripts/eval_pos.sh $alpha $train_alpha $vocab_size $src_lang $tgt_lang $tok_type $seed $probe
  done
done

# source run_pos_pipeline.sh 0.25 0.25 120000 1234