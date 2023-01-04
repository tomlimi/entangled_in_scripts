#!/bin/bash

cd /home/$USER/my-luster/entangled-in-scripts/entangled_in_scripts/scripts || exit 1;

alpha=0.25
train_alpha=0.25
seed=1234
probe=False # True or False
custom_head=True

echo "probe: $probe"
echo "custom_head: $custom_head"

vocab_size=120000
# model_type="bpe-tokenization"
# model_type="merged-tokenization"
# model_type="multilingual-tokenization"
model_type="20l-multilingual-tokenization"

# vocab_size=20000
# model_type="nooverlap-tokenization"

echo "model_type: $model_type"
echo "vocab_size: $vocab_size"

# langs=("ar" "el" "en" "es" "tr" "zh")
langs=("ar" "bg" "de" "el" "en" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh") # all XNLI languages
echo "langs: ${langs[@]}"

for lang_src in ${langs[@]}
do
    jid=$(sbatch finetune_xnli.sh $model_type $alpha $train_alpha $vocab_size $lang_src $seed $probe $custom_head --overwrite_output_dir)
    jid=${jid##* }
    echo $lang_src
    echo $jid
    for lang_tgt in ${langs[@]}
    do
        sbatch --dependency=afterany:$jid eval_xnli.sh $model_type $alpha $train_alpha $vocab_size $lang_src $lang_tgt $seed $probe $custom_head
    done

done

sleep 1s
tail -f ~/my-luster/entangled-in-scripts/job_outputs/xnli/finetune_$jid.out


# runs that crashed:
# sbatch eval_xnli.sh "multilingual-tokenization" 0.25 0.25 120000 es tr 8888 True