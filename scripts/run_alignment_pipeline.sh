#!/bin/bash
export LD_LIBRARY_PATH=/ha/home/limisiewicz/.virtualenvs/eis/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
cd /home/limisiewicz/my-luster/entangled-in-scripts/entangled_in_scripts || exit 1;

vocab_size=$1
alpha=$2
train_alpha=$3
tok_type=$4
root_path=$5

# langs=("ar" "el" "en" "es" "tr" "zh")
langs=("ar" "el" "en" "es" "tr" "zh" "sw" "hi" "mr" "ur" "ta" "te" "th" "ru" "bg" "he" "ka" "vi" "fr" "de")

for src_lang in ${langs[@]}
do
  echo $src_lang
  echo $seed
  echo $alpha
  echo $train_alpha
  echo $src_lang

for tgt_lang in ${langs[@]}
  do
  if [ "$src_lang" != "$tgt_lang" ] ; then
    sbatch scripts/eval_alignment.sh $alpha $train_alpha $vocab_size $src_lang $tgt_lang $tok_type $root_path
  fi
  done
done

# source run_alignment_pipeline.sh 120000 0.25 0.25 sp-unigram ..