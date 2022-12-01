#!/bin/bash

function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

alpha_id=$1
alpha_id_train=$2
vocab_size=$3
langs=${@:4:1000}

alphas=("0.0" "0.25" "0.5" "0.75" "1.0")

alpha=${alphas[@]:$alpha_id:1}
alpha_train=${alphas[@]:$alpha_id_train:1}

lang_string=$(join_by "-" ${langs[@]})
output_path="/lnet/work/people/limisiewicz/entangled-in-scripts/models/config/multilingual-tokenization"
tokenizer_path="/lnet/work/people/limisiewicz/entangled-in-scripts/tokenizers/sp-unigram/${lang_string}/alpha-${alpha}_N-${vocab_size}"
data_path="/lnet/express/work/people/limisiewicz/cc100"
seed=1234

#saving model config
MODEL_CONFIG=$( jq -n \
                  --arg tokenizer_path "$tokenizer_path" \
                  --argjson vs $vocab_size \
                  --argjson ml 128 \
                  --argjson hs 768 \
                  --argjson nl 8 \
                  --argjson nh 6 \
                  '{tokenizer_path: $tokenizer_path, hidden_layer_size: $hs, vocab_size: $vs, max_sent_len: $ml, num_hidden: $nl, num_attention: $nh}')


if [ -f "${output_path}/model_alpha-${alpha}_N-${vocab_size}.json" ]; then
  echo "${output_path}/model_alpha-${alpha}_N-${vocab_size}.json exists!"
  cat "${output_path}/model_alpha-${alpha}_N-${vocab_size}.json"
else
  echo $MODEL_CONFIG > "${output_path}/model_alpha-${alpha}_N-${vocab_size}.json"
  echo "${output_path}/model_alpha-${alpha}_N-${vocab_size}.json  created!"
fi

struct_train_config () {
  TRAIN_CONFIG=$( jq -n \
                --arg train "$arr_train" \
                --arg eval "$arr_eval" \
                --argjson ne 10 \
                --argjson bs 1024 \
                '{train_data_paths_list: $train | split(" "), eval_data_paths_list: $eval | split(" "), num_epochs: $ne, batch_size: $bs}')

  echo $TRAIN_CONFIG
}

# saving training configs

arr_eval=""
arr_train=""

for lang in ${langs[@]}
do
  arr_eval+="${data_path}/${lang}/dev "
  for alpha in ${alphas[@]:0:$alpha_id_train+1}
  do
    arr_train+="${data_path}/${lang}/alpha${alpha} "
  done
done



#echo $arr_eval0 | sed 's/^[ \t]*//;s/[ \t]*$//' > arr_eval

arr_train="$(sed -e 's/\ *$//g'<<<"${arr_train}")"
arr_eval="$(sed -e 's/\ *$//g'<<<"${arr_eval}")"

struct_train_config > "${output_path}/train_config_alpha-${alpha_train}.json"
echo "Saved: ${output_path}/${lang}/train_config_alpha-${alpha_train}.json"

#source prepare_mlm_training_configs.sh 1 1 120000 ar tr zh el es en


