#!/bin/bash

for alpha in 0 1 2 3 4; do
    for lang in ar tr zh el es en; do
        sbatch compute_token_frequency.sh 20000 $alpha sp-unigram $lang
    done
    # multilingual
    sbatch compute_token_frequency.sh 120000 $alpha sp-unigram ar tr zh el es en
done