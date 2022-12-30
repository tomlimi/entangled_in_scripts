#!/bin/bash
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --constraint="gpuram24G|gpuram40G|gpuram48G"
#SBATCH --time=59:00
#SBATCH --gres=gpu:1
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --output=/home/limisiewicz/my-luster/entangled-in-scripts/job_outputs/NER/eval_%j.out

cd /home/limisiewicz/my-luster/entangled-in-scripts/entangled_in_scripts || exit 1;
source /home/limisiewicz/my-luster/entangled-in-scripts/eis/bin/activate

alpha=$1
train_alpha=$2
vocab_size=$3
lang_src=$4
lang_tgt=$5
tok_type=$6
seed=$7
probe=$8


model_config="/home/limisiewicz/my-luster/entangled-in-scripts/models/config/${tok_type}-tokenization/model_alpha-${alpha}_N-${vocab_size}.json"
name="alpha-${alpha}_alpha-train-${train_alpha}_N-${vocab_size}"

#name="${name}_probe"

if [ "$probe" = 1 ]; then
    output_path="/home/limisiewicz/my-luster/entangled-in-scripts/models/NER_PROBE/${tok_type}-tokenization/"
else
    output_path="/home/limisiewicz/my-luster/entangled-in-scripts/models/NER_FT/${tok_type}-tokenization/"
fi

echo start...
echo NER
echo ${output_path}
echo ${model_config}
echo ${lang_src}_${lang_tgt}

#use --killable --requeue !!!
python src/eval_classification.py -o ${output_path} -p ${name} -s ${lang_src} -t ${lang_tgt} --model_config_path ${model_config} --seed ${seed} --ft_task NER --metric f1-macro

chmod -R 770 $output_path || exit 0;

echo end
