# source: https://towardsdatascience.com/pre-processing-a-wikipedia-dump-for-nlp-model-training-a-write-up-3b9176fdf67

"""
To pre-process the Wikipedia dump (extracted and cleaned), for example, simply run the following command in
your terminal: python3 preprocess_wiki_dump.py enwiki-latest-pages-articles.txt
"""

import os
from blingfire import text_to_sentences
import subprocess
import argparse
from tqdm import tqdm
from pathlib import Path
import lzma
from multiprocessing import Pool
from functools import partial

import constants

data_directory = "/lnet/express/work/people/limisiewicz/cc100"


def download_data(language_code, data_directory):
    # downloads data to specified repository
    print(language_code)
    if language_code == 'zh':
        download_code = 'zh-Hans'
    else:
        download_code = language_code
    try:
        subprocess.call(f"wget https://data.statmt.org/cc-100/{download_code}.txt.xz -O {data_directory}/{language_code}.txt.xz", shell=True)
        subprocess.call(f"mkdir {data_directory}/{language_code}", shell=True)
    except Exception as e:
        print("Error downloading {}, skipping".format(language_code))
        print(e.args)
        
        
def remove_data(language_code, data_directory):
    try:
        subprocess.call(f"rm -f {data_directory}/{language_code}.txt.xz", shell=True)
    except Exception as e:
        print("Error cleaning {}, skipping".format(language_code))
        print(e.args)


def process_data(language_code, data_directory):
    data_source = f"{data_directory}/{language_code}.txt.xz"
    target_directory = f"{data_directory}/{language_code}/"
    try:
        process(data_source, target_directory, constants.corpus_sizes[language_code])
    except Exception as e:
        print("Error preprocessing {}, skipping".format(language_code))
        print(e.args)


def process(path, output_dir, corpus_sizes):
    cc100_file_in = Path(path)
    print(cc100_file_in)
    print('Pre-processing {} to {}...'.format(cc100_file_in, output_dir))

    total_size = 0
    
    file_names_tmp = list(corpus_sizes.keys())
    data_lims_tmp = list(corpus_sizes.values())
    
    data_lims_tmp.append(data_lims_tmp[-1] + constants.val_test_size)
    file_names_tmp.append("_last")
    
    data_lim = data_lims_tmp.pop(0)
    file_name = file_names_tmp.pop(0)
    out_file_path = os.path.join(output_dir, file_name)
    
    with lzma.open(cc100_file_in, mode='rt', encoding='utf-8') as in_f:
        print("Reading lines...")
        o_f = open(out_file_path, "w")
        for line in tqdm(in_f):
            if len(line.split()) <= 2:
                continue
            if ('&lt' in line or '&gt' in line) and ';' in line:
                continue
            sentences = text_to_sentences(line)
            o_f.write(sentences + '\n')
            total_size += len(line)
            if total_size >= data_lim:
                o_f.close()
                if data_lims_tmp:
                    data_lim = data_lims_tmp.pop(0)
                    file_name = file_names_tmp.pop(0)
                else:
                    print('reached data limit.')
                    break

                out_file_path = os.path.join(output_dir,  file_name)
                o_f = open(out_file_path, "w")

    print('Successfully pre-processed {} to {}...'.format(cc100_file_in,
                                                          output_dir))


def main(language_code, data_directory):
    if args.download:
        download_data(language_code, data_directory)

    process_data(language_code, data_directory)
    
    if args.remove:
        remove_data(language_code, data_directory)

#
# def main_multiprocess(args):
#     language_codes = args.language_codes
#     with Pool(processes=args.multiprocess) as pool:
#         if args.download:
#             pool.map(partial(download_data, data_directory= args.output_directory), language_codes)
#
#         pool.map(partial(process_data, data_directory= args.output_directory),language_codes)
#
#         if args.remove:
#             pool.map(partial(remove_data, data_directory= args.output_directory), language_codes)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--language_codes', nargs='+', help='<Required>, list of language codes to download and/or parse', required=True)
    parser.add_argument('-o', '--output_directory', type=str, default="/lnet/express/work/people/limisiewicz/cc100")
    parser.add_argument('-d','--download', type=bool, default=True)
    parser.add_argument('-r','--remove', type=bool, default=True)
    parser.add_argument('-m', '--multiprocess', type=int, default=0)
    args = parser.parse_args()
    
    if args.multiprocess:
        with Pool(processes=args.multiprocess) as pool:
            pool.map(partial(main, data_directory=args.output_directory), args.language_codes)
    else:
        for language_code in args.language_codes:
            main(language_code, args.output_directory)
    # -l ar tr zh el es en
    # -l sw