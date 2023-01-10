#!/bin/bash

cd /home/$USER/my-luster/entangled-in-scripts/entangled_in_scripts/scripts || exit 1;

alpha=0.25
train_alpha=0.25
probe=True # True or False
custom_head=True
epochs=30

echo "probe: $probe"
echo "custom_head: $custom_head"

vocab_size=120000
langs=("ar" "el" "en" "es" "tr" "zh")
model_type="multilingual-tokenization"
# model_type="merged-tokenization"
# model_type="bpe-tokenization"

# vocab_size=20000
# model_type="nooverlap-tokenization"

# vocab_size=120000
# langs=("ar" "bg" "de" "el" "en" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh") # all XNLI languages
# model_type="20l-multilingual-tokenization"
# model_type="20l-merged-tokenization"
# model_type="20l-multilingual-tokenization"

echo "model_type: $model_type"
echo "vocab_size: $vocab_size"

echo "langs: ${langs[@]}"
for lang_src in ${langs[@]}
do
    jid=0
    for seed in "2000" "2001" "2002" "2003" "2004"
    do
        jid=$(sbatch finetune_xnli.sh $model_type $alpha $train_alpha $vocab_size $lang_src $seed $probe $custom_head --overwrite_output_dir --precompute_model_outputs --num_train_epochs $epochs)
        jid=${jid##* }
        echo $lang_src
        echo $jid
        for lang_tgt in ${langs[@]}
        do
            # sbatch eval_xnli.sh $model_type $alpha $train_alpha $vocab_size $lang_src $lang_tgt $seed $probe $custom_head
            sbatch --dependency=afterany:$jid eval_xnli.sh $model_type $alpha $train_alpha $vocab_size $lang_src $lang_tgt $seed $probe $custom_head
        done
    done
done

# sleep 1s
# tail -f ~/my-luster/entangled-in-scripts/job_outputs/xnli/finetune_$jid.out


# runs that crashed:
# sbatch eval_xnli.sh "multilingual-tokenization" 0.25 0.25 120000 es tr 8888 True