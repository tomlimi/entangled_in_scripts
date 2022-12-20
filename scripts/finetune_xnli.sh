#!/bin/bash
#SBATCH --mem=8g
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --time=59:00
#SBATCH --gres=gpu:1
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=balhar.j@gmail.com
#SBATCH --output=/home/balhar/my-luster/entangled-in-scripts/job_outputs/xnli/finetune_%j.out

cd /home/$USER/my-luster/entangled-in-scripts/entangled_in_scripts || exit 1;
source /home/$USER/my-luster/entangled-in-scripts/eis/bin/activate

alpha=$1
train_alpha=$2
vocab_size=$3
lang=$4
seed=$5

in_seed=1234

input_path="/home/$USER/my-luster/entangled-in-scripts/models/LM/multilingual-tokenization"
name="alpha-${alpha}_alpha-train-${train_alpha}_N-${vocab_size}"
model_path="$input_path/${name}_${in_seed}"


# extract tokenizer path from the model_config json file
model_config="/home/$USER/my-luster/entangled-in-scripts/models/config/multilingual-tokenization/model_alpha-${alpha}_N-${vocab_size}.json"
tokenizer_path=$(python -c "import json; print(json.load(open('$model_config'))['tokenizer_path'])")

output_path="/home/$USER/my-luster/entangled-in-scripts/models/XNLI_FT/multilingual-tokenization/"
ft_output_path="$output_path/${name}_$seed/$lang"

eval_and_save_steps=100

echo start...
echo XNLI
echo ${input_path}
echo ${output_path}
echo ${model_config}
echo ${tokenizer_path}
echo ${name}

# disable cuda
# export CUDA_VISIBLE_DEVICES=""
# python -m pdb src/finetune_xnli.py \

python src/finetune_xnli.py \
    --model_name_or_path ${model_path} --tokenizer_name ${tokenizer_path} --output_dir ${ft_output_path} --seed ${seed} --train_language ${lang} --language ${lang} \
    --max_seq_length 126 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --save_steps $eval_and_save_steps --eval_steps $eval_and_save_steps \
    --save_total_limit 5 --learning_rate 2e-5 --weight_decay 0.01 --evaluation_strategy steps --do_train --do_eval


chmod -R 770 $output_path || exit 0;

echo end

# Example:
# bash finetune_xnli.sh 0.25 0.25 120000 en 333