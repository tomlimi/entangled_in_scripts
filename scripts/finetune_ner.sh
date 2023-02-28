#!/bin/bash
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --time=23:59:00
#SBATCH --constraint="gpuram24G|gpuram40G|gpuram48G"
#SBATCH --gres=gpu:1
#SBATCH -p gpu-troja,gpu-ms

cd /home/limisiewicz/my-luster/entangled-in-scripts/entangled_in_scripts || exit 1;
source /home/limisiewicz/my-luster/entangled-in-scripts/eis/bin/activate


vocab_size=$1
alpha=$2
train_alpha=$3
lang=$4
tok_type=$5
seed=$6
probe=$7
root_path=$8


input_path="${root_path}/models/LM/${tok_type}-tokenization"
model_config="${root_path}/models/config/$tok_type-tokenization/model_alpha-${alpha}_N-${vocab_size}.json"
name="alpha-${alpha}_alpha-train-${train_alpha}_N-${vocab_size}"

#name="${name}_probe"

if [ "$probe" = 1 ]; then
    output_path="${root_path}/models/NER_PROBE/${tok_type}-tokenization/"
else
    output_path="${root_path}/models/NER_FT/${tok_type}-tokenization/"
fi




echo start...
echo NER
echo ${input_path}
echo ${output_path}
echo ${model_config}
echo ${name}

#use --killable --requeue !!!
if [ "$probe" = 1 ]; then
    python src/finetune_classification.py -i ${input_path} -o ${output_path} -p ${name} -l ${lang} --model_config_path ${model_config} --seed ${seed} --ft_task NER --probe True
else
    python src/finetune_classification.py -i ${input_path} -o ${output_path} -p ${name} -l ${lang} --model_config_path ${model_config} --seed ${seed} --ft_task NER --probe False
fi
chmod -R 770 $output_path || exit 0;

echo end

#sbatch finetune_ner.sh 120000 0.25 0.25 ar sp-unigram 1234 0 ../..
