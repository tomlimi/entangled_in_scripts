#!/bin/bash

cd /home/$USER/my-luster/entangled-in-scripts/entangled_in_scripts/scripts || exit 1;

seeds=("2000" "2001" "2002" "2003" "2004")

vocab_size=120000
langs=("ar" "el" "en" "es" "tr" "zh")
model_type="multilingual"
# model_type="merged"
# model_type="bpe"

# vocab_size=20000
# model_type="nooverlap"

# vocab_size=120000
# langs=("ar" "el" "en" "es" "tr" "zh" "sw" "hi" "mr" "ur" "ta" "te" "ru" "bg" "he" "vi" "fr" "de") # all UD languages
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
        echo "lang src $lang_src, seed $seed"
        jid=$(sbatch finetune_ud.sh $model_type $vocab_size $lang_src $seed)
        jid=${jid##* }
        echo $jid
        eval_jids=""
        for lang_tgt in ${langs[@]}
        do
            echo eval_ud.sh $model_type $vocab_size $lang_src $lang_tgt $seed
            eval_jid=$(sbatch --dependency=afterok:$jid eval_ud.sh $model_type $vocab_size $lang_src $lang_tgt $seed)
            eval_jid=${eval_jid##* }
            eval_jids="$eval_jids:$eval_jid"
        done
        echo $eval_jids
        clean_jid=$(sbatch --dependency=afterok$eval_jids clean_ft_files_per_lang.sh 0.25 0.25 $vocab_size $model_type "UD" $seed 1 $lang_src)
    done
done

# after evaluation, clean up the model checkpoints:
# find /home/balhar/my-luster/entangled-in-scripts/models/UD_PROBE/ -name "pytorch_model.bin" -delete

# fix german and russian
# ls UD_PROBE/20l*/*_2000/*/accuracy_evaluation/de/accuracy_all.txt
# ls UD_PROBE/20l*/*_2000/de
