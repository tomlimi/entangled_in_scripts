#!/bin/bash
alpha="$1"
langs=${@:2:1000}

data_file="/lnet/express/work/people/limisiewicz/cc100"
alphas=("0.0" "0.25" "0.5" "0.75" "1.0")

# get the alpha_id by finding the index of the alpha in the alphas array
alpha_id=-1
for i in "${!alphas[@]}"; do
    if [[ "${alphas[$i]}" = "${alpha}" ]]; then
        alpha_id=$i
        break
    fi
done
# if the alpha is not in the array, exit with error
if [[ $alpha_id -eq -1 ]]; then
    echo "Alpha ${alpha} not found in the array of alphas: ${alphas[@]}"
    exit 1
fi

for lang in ${langs[@]}
do
  for alpha in ${alphas[@]:0:$alpha_id+1}
  do
    new_path="${data_file}/${lang}/alpha${alpha}"
    # check if dir or symlink exists
    if [ ! -e $new_path ]; then
        echo "Path ${new_path} does not exist."
        exit 1
    fi
    files+="$new_path "
  done
done

echo $files
