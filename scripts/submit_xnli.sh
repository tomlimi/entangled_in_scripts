#!/bin/bash

cd /home/$USER/my-luster/entangled-in-scripts/entangled_in_scripts/scripts || exit 1;

alpha=0.25
train_alpha=0.25
vocab_size=120000
seed=8888
lang_src=en
probe=True # True or False

for lang_src in "es" # 'ar' 'el' 'es' 'tr' 'zh'
do
    jid=$(sbatch finetune_xnli.sh $alpha $train_alpha $vocab_size $lang_src $seed $probe)
    jid=${jid##* }
    echo $lang_src
    echo $jid
    for lang_tgt in 'ar' 'el' 'en' 'es' 'tr' 'zh'
    do
        sbatch --dependency=afterany:$jid eval_xnli.sh $alpha $train_alpha $vocab_size $lang_src $lang_tgt $seed $probe
    done

done

# sleep 1s
# tail -f ~/my-luster/entangled-in-scripts/job_outputs/xnli/finetune_$jid.out
