#!/bin/bash
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=3-0
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=balhar.j@gmail.com
#SBATCH --output=/home/limisiewicz/my-luster/entangled-in-scripts/job_outputs/compute_frequency_%j.out

# arguments for loading the correct tokenizer
vocab_size=$1
alpha=$2
type=$3
tokenizer_lang=$4
experiment_name=$5
# arguments for loading the data
data_args=${@:6:1000}

cd /home/$USER/my-luster/entangled-in-scripts/entangled_in_scripts/src || exit 1;
source /home/limisiewicz/my-luster/entangled-in-scripts/eis/bin/activate

output_path="/lnet/work/people/limisiewicz/entangled-in-scripts/tokenizers"

echo "Data arguments: ${data_args}"
echo "Type: ${type}"
echo "Alpha: ${alpha}"
echo "Vocab size: ${vocab_size}"
echo "Langs: ${langs[@]}"
echo "Cased: yes"
echo "Name: $experiment_name"

# python compute_token_frequency.py -o ${output_path} -a $alpha -l ${langs[@]} -v $vocab_size -t $type -c
# identifikace tokenizeru: tokenizer_dir, tokenizer_type, lang, alpha, NV
python compute_token_frequency.py -o ${output_path} -t $type -l $tokenizer_lang -a $alpha -v $vocab_size -n $experiment_name -c ${data_args}

# Run:
# sbatch compute_token_frequency.sh 120000 $alpha sp-unigram ar-tr-zh-el-es-en "token_freq_$lang_$alpha" -d $(./pretraining_data_paths.sh $alpha $lang)
