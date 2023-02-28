#!/bin/bash
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --constraint="gpuram24G|gpuram40G|gpuram48G"
#SBATCH --time=59:00
#SBATCH --gres=gpu:1
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --output=/home/limisiewicz/my-luster/entangled-in-scripts/job_outputs/POS/eval_%j.out

cd .. || exit 1;
source ../../eis/bin/activate

alpha=$1
train_alpha=$2
vocab_size=$3
lang_src=$4
lang_tgt=$5
tok_type=$6
seed=$7
probe=$8
root_path=$9


model_config="${root_path}/models/config/${tok_type}-tokenization/model_alpha-${alpha}_N-${vocab_size}.json"
name="alpha-${alpha}_alpha-train-${train_alpha}_N-${vocab_size}"

#name="${name}_probe"

if [ "$probe" = 1 ]; then
    output_path="${root_path}/models/POS_PROBE/${tok_type}-tokenization/"
else
    output_path="${root_path}/models/POS_FT/${tok_type}-tokenization/"
fi

#name="${pt_type}_${vocab}_depth_${depth}"


echo start...
echo POS
echo ${output_path}
echo ${model_config}
echo ${lang_src}_${lang_tgt}

#use --killable --requeue !!!
python src/eval_classification.py -o ${output_path} -p ${name} -s ${lang_src} -t ${lang_tgt} --model_config_path ${model_config} --seed ${seed} --ft_task POS
python src/eval_classification.py -o ${output_path} -p ${name} -s ${lang_src} -t ${lang_tgt} --model_config_path ${model_config} --seed ${seed} --ft_task POS --metric f1-macro

chmod -R 770 $output_path || exit 0;

echo end

# sbatch eval_pos.sh 0.25 0.25 120000 ar en sp-unigram 1234 0 ../..