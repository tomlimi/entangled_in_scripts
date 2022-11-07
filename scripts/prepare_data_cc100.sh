#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c4
#SBATCH --time=3-0
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=limisiewicz@ufal.mff.cuni.cz
#SBATCH --output=/home/limisiewicz/my-luster/entangled-in-scripts/job_outputs/data_process_cc110_%j.out


cd /home/limisiewicz/my-luster/entangled-in-scripts/entangled_in_scripts/src || exit 1;
source /home/limisiewicz/my-luster/entangled-in-scripts/eis/bin/activate

data_file="/lnet/express/work/people/limisiewicz/cc100"

langs=("ar" "tr" "zh" "el" "es" "en")
#langs=("sw" "ar")

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

