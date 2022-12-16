## Entangled in the Scripts

We check which factors (size of voacnulary, language set, balance between languages) affect multilingual tokenization in LMs.

Then, we analyze how multilingual tokenization (share of each language in multilingual voabulary, overlap of vocabularies of specific languages) affect performance on down-stream tasks and  cross-lingual transfer.

## Preliminaries

Install the requirements:
```
pip install -r entangled_in_scipts/requirements.txt
```

Open the working directory with scripts
```
cd entangled_in_scripts/scripts
```


## Downloading data

For experiments we use portion of the CC100 dataset (for six languages: Arabic, Turkish, Chinese, Greek, Spanish, English) and the UD treebanks (for 6 languages: Arabic, Turkish, Chinese, Greek, Spanish, English).
To download the data run:
```
source prepare_data_cc100.sh <path_to_data_dir>
```

The script will download the data and create the following directory structure:
```
<path_to_data_dir>
├── lang
│   ├── alpha0.0
│   ├── alpha0.25
│   ├── alpha0.5
│   ├── alpha0.75
│   ├── alpha1.0
│   ├── dev
│   ├── test
```
alpha[0.0, 0.25, 0.5, 0.75, 1.0] are text files with accumulated data from CC100 dataset (for each language separately). 
The size of the text per file is defined by the equation:


$`c_L = c_{min} \cdot (\frac{|C_L|}{c_{min}})^\alpha`$

where $`c_{min}`$ is the minimal size of the text file, $`|C_L|`$ is the maximal size for language $`L`$ . $`\alpha`$ is the parameter that controls the sized of the corpus and balance between languages. (e.g. for $`\alpha=0.0`$ size of data is equal for all languages).   

dev, test are development and test sets for each language.

### Training tokenizers

To train tokenizer run:
```
source train_tokenizer.sh <vocab_size> <alpha_idx> <type> <langugage_list>
```

<vocab_size> is the size of the vocabulary for the tokenizer.
<alpha_idx> is the index of the alpha parameter in the list [0.0, 0.25, 0.5, 0.75, 1.0] defining how many files should be accumulated per language.
<type> currently supporteted types of the tokenizer are bpe, unigram, sp-bpe, sp-unigram (where sp stands for SentencePiece method).


### Training LMs


### Fine tuning models on downstream tasks