#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c4
#SBATCH --time=3-0


cd ../src || exit 1;
# source ../../eis/bin/activate

data_file=$1

langs=("sw" "hi" "mr" "ur" "ta" "te" "th" "ru" "bg" "he" "ka" "vi" "fr" "de")

python data_generator_cc100.py -l ${langs[@]} -o $data_file -d True -r True -m 3

for lang in ${langs[@]}
do
  echo "Preparing test dev splits in $lang"
  cat "${data_file}/${lang}/_last"  | shuf  > "${data_file}/${lang}/test_dev"
  split -n l/2 "${data_file}/${lang}/test_dev" "${data_file}/${lang}/test_dev."
  mv "${data_file}/${lang}/test_dev.aa" "${data_file}/${lang}/dev"
  mv "${data_file}/${lang}/test_dev.ab" "${data_file}/${lang}/test"
  rm "${data_file}/${lang}/test_dev"
done

chmod -R 777 ${data_file}

# sbatch prepare_data_cc100.sh "../../data/cc100"