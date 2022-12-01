#!/bin/bash
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=3-0
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=limisiewicz@ufal.mff.cuni.cz
#SBATCH --output=/home/limisiewicz/my-luster/entangled-in-scripts/job_outputs/tokenizer_train_%j.out

#langs="hi he ru ko el ur te en de hu eu vi tr es"
#langs="hi he ru ko el ur te ug en de hu eu vi tr es csb"

vocab_size=$1
alpha_id=$2
type=$3
langs=${@:4:1000}

alphas=("0.0" "0.25" "0.5" "0.75" "1.0")

cd /home/limisiewicz/my-luster/entangled-in-scripts/entangled_in_scripts/src || exit 1;
source /home/limisiewicz/my-luster/entangled-in-scripts/eis/bin/activate

data_file="/lnet/express/work/people/limisiewicz/cc100"
output_path="/lnet/work/people/limisiewicz/entangled-in-scripts/tokenizers"
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

#sbatch train_tokenizer.py 1 sp-unigram ar tr zh el es en