#!/bin/bash
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --output=/home/limisiewicz/my-luster/entangled-in-scripts/job_outputs/ALIGN/eval_%j.out

cd .. || exit 1;
source ../../eis/bin/activate

vocab_size=$1
alpha=$2
train_alpha=$3
lang_src=$4
lang_tgt=$5
tok_type=$6
root_path=$7

input_path="${root_path}/models/LM/${tok_type}-tokenization"
model_config="${root_path}/models/config/${tok_type}-tokenization/model_alpha-${alpha}_N-${vocab_size}.json"
name="alpha-${alpha}_alpha-train-${train_alpha}_N-${vocab_size}"

#name="${name}_probe"

output_path="${root_path}/models/ALIGN/${tok_type}-tokenization/"


echo start...
echo ALIGN
echo ${input_path}
echo ${output_path}
echo ${model_config}
echo ${lang_src}_${lang_tgt}

python src/eval_alignment.py -o ${output_path} -i ${input_path} -p ${name} -s ${lang_src} -t ${lang_tgt} --model_config_path ${model_config} --seed 1234

chmod -R 770 $output_path || exit 0;

echo end

# sbatch eval_alignment.sh 120000 0.25 0.25 ar en sp-unigram ../..