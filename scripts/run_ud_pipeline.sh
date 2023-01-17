#!/bin/bash

cd /home/$USER/my-luster/entangled-in-scripts/entangled_in_scripts/scripts || exit 1;

seeds=("2000" "2001" "2002" "2003" "2004")

vocab_size=120000
langs=("ar" "el" "en" "es" "tr" "zh")
# model_type="multilingual"
# model_type="merged"
# model_type="bpe"

vocab_size=20000
model_type="nooverlap"

# TODO: change to UD
# vocab_size=120000
# langs=("ar" "bg" "de" "el" "en" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh") # all XNLI languages
# langs=("ar") # all XNLI languages
# model_type="20l-multilingual"
# model_type="20l-merged"
# model_type="20l-bpe"

echo "model_type: $model_type"
echo "vocab_size: $vocab_size"
echo "langs: ${langs[@]}"
echo "seeds: ${seeds[@]}"

for lang_src in ${langs[@]}
do
    for seed in ${seeds[@]}
    do
        echo "lang src $lang_src"
        echo "seed $seed"
        jid=$(sbatch finetune_ud.sh $model_type $vocab_size $lang_src $seed)
        jid=${jid##* }
        echo $jid
        for lang_tgt in ${langs[@]}
        do
            sbatch --dependency=afterany:$jid eval_ud.sh $model_type $vocab_size $lang_src $lang_tgt $seed
        done
    done
done

# after evaluation, clean up the model checkpoints:
# find /home/balhar/my-luster/entangled-in-scripts/models/UD_PROBE/ -name "pytorch_model.bin" -delete