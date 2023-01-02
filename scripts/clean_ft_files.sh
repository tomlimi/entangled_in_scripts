#!/bin/bash
#SBATCH --mem=8g
#SBATCH -N 1
#SBATCH --time=59:00
#SBATCH --output=/home/limisiewicz/my-luster/entangled-in-scripts/job_outputs/clean_ft_files_%j.out

alpha=$1
train_alpha=$2
vocab_size=$3
tok_type=$4
task=$5
seed=$6
probe=$7


model_name="alpha-${alpha}_alpha-train-${train_alpha}_N-${vocab_size}_${seed}"
if [ "$probe" = 1 ]; then
    output_path="/home/limisiewicz/my-luster/entangled-in-scripts/models/${task}_PROBE/${tok_type}-tokenization/${model_name}"
else
    output_path="/home/limisiewicz/my-luster/entangled-in-scripts/models/${task}_FT/${tok_type}-tokenization/${model_name}"
fi

cd $output_path  || exit 1;

rm */pytorch_model.bin
rm -r */checkpoint-*

echo "cleaned $output_path"