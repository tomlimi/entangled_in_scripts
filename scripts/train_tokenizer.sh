#!/bin/bash
#SBATCH --mem=64g
#SBATCH -c4
#SBATCH --time=3-0

vocab_size=$1
alpha_id=$2
type=$3
data_file=$4
output_path=$5
langs=${@:6:1000}

alphas=("0.0" "0.25" "0.5" "0.75" "1.0")

cd ../src || exit 1;
# source ../../eis/bin/activate

files=""

for lang in ${langs[@]}
do
  for alpha in ${alphas[@]:0:$alpha_id+1}
  do
    files+="${data_file}/${lang}/alpha${alpha} "
  done
done

alpha=${alphas[@]:$alpha_id:1}

echo "Tokenizer training files: ${files}"
echo "Type: ${type}"
echo "Alpha: ${alpha}"
echo "Vocab size: ${vocab_size}"
echo ""

python train_tokenizer.py -d ${files} -o ${output_path} -a $alpha -l ${langs[@]} -v $vocab_size -t $type -c True

# sbatch train_tokenizer.py 1 sp-unigram "../../data/cc100" "../../tokenizers" ar tr zh el es en