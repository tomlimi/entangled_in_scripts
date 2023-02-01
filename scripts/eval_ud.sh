#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=balhar.j@gmail.com
#SBATCH --output=/home/balhar/my-luster/entangled-in-scripts/job_outputs/UD/eval_%j.out

set -eu

PROJECT_DIR=/home/$USER/my-luster/entangled-in-scripts

cd $PROJECT_DIR/entangled_in_scripts || exit 1;
source $PROJECT_DIR/eis/bin/activate

tok_type=$1
vocab_size=$2
lang_src=$3
lang_tgt=$4
seed=$5

alpha="0.25"
train_alpha="0.25"
seed_in=1234

model_config_path="/home/limisiewicz/my-luster/entangled-in-scripts/models/config/$tok_type-tokenization/model_alpha-${alpha}_N-${vocab_size}.json"
name="alpha-${alpha}_alpha-train-${train_alpha}_N-${vocab_size}"

output_path="$PROJECT_DIR/models/UD_PROBE/${tok_type}-tokenization/"

ft_output_path="$output_path/${name}_${seed}/$lang_src"

if [ -f "$ft_output_path/accuracy_evaluation/$lang_tgt/accuracy_all.txt" ]; then
    echo "model $ft_output_path already evaluated on $lang_tgt; skipping..."
    exit 0;
fi

echo start...
echo UD
echo "tok_type $tok_type"
echo "vocab_size $vocab_size"
echo "lang_src $lang_src"
echo "lang_tgt $lang_tgt"
echo "seed $seed"
echo ""
echo "ft_output_path $ft_output_path"
echo "model_config_path $model_config_path"
echo ""
# export CUDA_VISIBLE_DEVICES=""

export TOKENIZERS_PARALLELISM=false

# dry run: --max_test_samples 1280
python src/finetune_ud.py \
    --pt_input_path $ft_output_path --ft_output_path $ft_output_path --model_config_path $model_config_path \
    --language $lang_tgt --seed $seed \
    --do_predict \
    --probe


chmod -R 770 $output_path || exit 0;

echo end
