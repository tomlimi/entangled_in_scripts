#!/bin/bash

for alpha in "0.0" "0.25" "0.5" "0.75" "1.0"; do
    # multilingual
    for lang in "ar" "tr" "zh" "el" "es" "en"; do
        sbatch compute_token_frequency.sh 120000 $alpha sp-unigram ar-tr-zh-el-es-en "token_freq_${lang}_${alpha}" -d $(./pretraining_data_paths.sh $alpha $lang)
    done
done
