#!/bin/bash
#SBATCH --mem=8g
#SBATCH -c2
#SBATCH --time=59:00
#SBATCH --gres=gpu:1
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --output=/home/limisiewicz/my-luster/entangled-in-scripts/job_outputs/POS/ft_%j.out

cd /home/limisiewicz/my-luster/entangled-in-scripts/entangled_in_scripts || exit 1;
source /home/limisiewicz/my-luster/entangled-in-scripts/eis/bin/activate



alpha=$1
train_alpha=$2
vocab_size=$3
lang=$4
seed=$5


input_path="/home/limisiewicz/my-luster/entangled-in-scripts/models/LM/multilingual-tokenization"
model_config="/home/limisiewicz/my-luster/entangled-in-scripts/models/config/multilingual-tokenization/model_alpha-${alpha}_N-${vocab_size}.json"
name="alpha-${alpha}_alpha-train-${train_alpha}_N-${vocab_size}"

#name="${name}_probe"


output_path="/home/limisiewicz/my-luster/entangled-in-scripts/models/POS_FT/multilingual-tokenization/"



echo start...
echo POS
echo ${input_path}
echo ${output_path}
echo ${model_config}
echo ${name}

#use --killable --requeue !!!
python src/finetune_classification.py -o ${output_path} -i ${input_path} -p ${name} -l ${lang} --model_config_path ${model_config} --load_checkpoint True --seed ${seed} --ft_task POS --probe False

chmod -R 770 $output_path || exit 0;

echo end
