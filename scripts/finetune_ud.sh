#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c2
#SBATCH --time=59:00
#SBATCH --constraint="gpuram24G|gpuram40G|gpuram48G"
#SBATCH --gres=gpu:1
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --output=/home/balhar/my-luster/entangled-in-scripts/job_outputs/UD/finetune_%j.out

set -eu

mkdir -p /home/$USER/my-luster/entangled-in-scripts/job_outputs/UD/
PROJECT_DIR=/home/$USER/my-luster/entangled-in-scripts

cd $PROJECT_DIR/entangled_in_scripts || exit 1;
source $PROJECT_DIR/eis/bin/activate

tok_type=$1
vocab_size=$2
lang=$3
seed=$4

alpha="0.25"
train_alpha="0.25"
seed_in=1234

input_path="/home/limisiewicz/my-luster/entangled-in-scripts/models/LM/${tok_type}-tokenization"
model_config_path="/home/limisiewicz/my-luster/entangled-in-scripts/models/config/$tok_type-tokenization/model_alpha-${alpha}_N-${vocab_size}.json"
name="alpha-${alpha}_alpha-train-${train_alpha}_N-${vocab_size}"

output_path="$PROJECT_DIR/models/UD_PROBE/${tok_type}-tokenization/"

pt_input_path="$input_path/${name}_${seed_in}"
ft_output_path="$output_path/${name}_${seed}/$lang"

echo start...
echo UD
echo "pt_input_path $pt_input_path"
echo "ft_output_path $ft_output_path"
echo "model_config_path $model_config_path"

export TOKENIZERS_PARALLELISM=false

python src/finetune_ud.py \
    --pt_input_path $pt_input_path --ft_output_path $ft_output_path --model_config_path $model_config_path \
    --language $lang --seed $seed \
    --truncate_at 10 --eval_and_save_steps 100 \
    --probe

chmod -R 770 $output_path || exit 0;

echo end

# ./finetune_ud.sh multilingual 120000 en 1234
