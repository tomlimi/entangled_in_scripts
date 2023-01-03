#!/bin/bash

cd /home/$USER/my-luster/entangled-in-scripts/entangled_in_scripts/scripts || exit 1;

alpha=0.25
train_alpha=0.25
seed=1234
probe=False # True or False

# vocab_size=120000
# model_type="bpe-tokenization"

vocab_size=20000
model_type="nooverlap-tokenization"

for lang_src in 'ar' 'el' 'es' 'en' 'tr' 'zh'
do
    jid=$(sbatch finetune_xnli.sh $model_type $alpha $train_alpha $vocab_size $lang_src $seed $probe)
    jid=${jid##* }
    echo $lang_src
    echo $jid
    for lang_tgt in 'ar' 'el' 'en' 'es' 'tr' 'zh'
    do
        sbatch --dependency=afterany:$jid eval_xnli.sh $model_type $alpha $train_alpha $vocab_size $lang_src $lang_tgt $seed $probe
    done

done

# sleep 1s
# tail -f ~/my-luster/entangled-in-scripts/job_outputs/xnli/finetune_$jid.out


# runs that crashed:
# sbatch eval_xnli.sh "multilingual-tokenization" 0.25 0.25 120000 es tr 8888 True