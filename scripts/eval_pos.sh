#!/bin/bash
#SBATCH --mem=8g
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --time=59:00
#SBATCH --gres=gpu:1
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --output=/home/limisiewicz/my-luster/entangled-in-scripts/job_outputs/POS/eval_%j.out

cd /home/limisiewicz/my-luster/entangled-in-scripts/entangled_in_scripts || exit 1;
source /home/limisiewicz/my-luster/entangled-in-scripts/eis/bin/activate

alpha=$1
train_alpha=$2
vocab_size=$3
lang_src=$4
lang_tgt=$5
tok_type=$6
seed=$7


model_config="/home/limisiewicz/my-luster/entangled-in-scripts/models/config/${tok_type}-tokenization/model_alpha-${alpha}_N-${vocab_size}.json"
name="alpha-${alpha}_alpha-train-${train_alpha}_N-${vocab_size}"

#name="${name}_probe"


output_path="/home/limisiewicz/my-luster/entangled-in-scripts/models/POS_PROBE/${tok_type}-tokenization/"
#name="${pt_type}_${vocab}_depth_${depth}"


echo start...
echo NER
echo ${output_path}
echo ${model_config}
echo ${lang_src}_${lang_tgt}

#use --killable --requeue !!!
python src/eval_classification.py -o ${output_path} -p ${name} -s ${lang_src} -t ${lang_tgt} --model_config_path ${model_config} --seed ${seed} --ft_task POS

chmod -R 770 $output_path || exit 0;

echo end
