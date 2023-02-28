#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram24G|gpuram40G|gpuram48G"
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=balhar.j@gmail.com
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

# if the file train_results.json exists in ft_output_path, then the model has already been trained
if [ -f "$ft_output_path/train_results.json" ]; then
    echo "model $ft_output_path already trained; skipping..."
    exit 0;
fi

echo start...
echo UD
echo "tok_type $tok_type"
echo "vocab_size $vocab_size"
echo "lang $lang"
echo "seed $seed"
echo ""
echo "pt_input_path $pt_input_path"
echo "ft_output_path $ft_output_path"
echo "model_config_path $model_config_path"
echo ""

# export CUDA_VISIBLE_DEVICES=""

export TOKENIZERS_PARALLELISM=false

# dry run: --max_train_samples 1280 --max_eval_samples 1280 --eval_and_save_steps 200
    # --max_train_samples 1280 --max_eval_samples 1280 --eval_and_save_steps 200 \
python src/finetune_ud.py \
    --pt_input_path $pt_input_path --ft_output_path $ft_output_path --model_config_path $model_config_path \
    --language $lang --seed $seed \
    --eval_and_save_steps 5000 \
    --do_train --do_eval \
    --probe

chmod -R 770 $output_path || exit 0;

echo end

# ./finetune_ud.sh multilingual 120000 en 1234
