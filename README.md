## Entangled in the Scripts

The code for the paper "Tokenization Impacts Multilingual Language Modeling: Assessing Vocabulary Allocation and Overlap Across Languages" by Tomasz Limisiewicz, Jiří Balhar, and David Mareček, and (two first authors developed the code).


In the paper, we discuss the importance of multilingual tokenizer properties (vocabulary allocation and cross-lingual
overlap) on LM performance in maksed language modeling and downstream tasks. 
We release our code for evaluating the multilingual tokenizers' key properties. 

Furthermore, we show code walkthrough that include steps to reproduce the experiments from the paper. 
Specifically, we show how to:
- Download the data (CC100 sample)
- Train tokenizers (BPE, Unigram, SentencePiece BPE, SentencePiece Unigram)
- Train mulitlingual LMs and evaluate them on masked language modeling task
- Fine-tune and evaluate the models on downstream tasks (NER, POS, Dependency labeling, XNLI)

## Preliminaries

Install the requirements:
```
pip install -r entangled_in_scipts/requirements.txt
```

Open the working directory with scripts
```
cd entangled_in_scripts/scripts
```

## Evaluate the tokenizer

The script `evaluate_tokenizer.py` loads tokenizer (pre-saved or from hugging face) and evaluates on metrics described in the paper:
- Vocabulary Overlap measured by Jensen-Shannon divergence between in-language distributions;
- Vocabulary Allocation measured by average rank of the token in in-language distribution.
- Vocabulary Allocation measured by the average number of characters for a token in specific language.
- Coverage, i.e. 1 - the share of unkown tokens in the tokenized text.

The results are saved as a json file `tokenizer_properties.json`. To run the evaluation run the following command (with exemplary parameters):

```bash
python evaluate_tokenizer.py \
    --data_list data_en.txt data_en2.txt data_es.txt data_pl.txt \
    --languages en en es pl \
    --tokenizer_name xlm-roberta-base \
    --output_dir /home/tokenizers_evaluation \
    [--unk_token <unk>]
```

The explanation of the parameters:
- data_list: listed paths to data
- languages: listed languages of the data for each data path
- tokenizer_name: HF tokenizer name or name of the tokenizer pre-saved in the output directory
- output_dir: path to output directory to save the results
- unk_token: optional, the unkonwn token in the vocabulary (by default `<unk>`)


## Reproducing the experiments

### Downloading data

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
source train_tokenizer.sh <vocab_size> <alpha_idx> <tokenizer_type> <path_to_data_dir> <output_path> <langugage_list>
```

<vocab_size> is the size of the vocabulary for the tokenizer.

<alpha_idx> is the index of the alpha parameter in the list [0.0, 0.25, 0.5, 0.75, 1.0] defining how many files should be accumulated per language.

<tokenizer_type> currently supporteted types of the tokenizer are bpe, unigram, sp-bpe, sp-unigram (where sp stands for SentencePiece method).

<path_to_data_dir> is the path to the directory with data, same as above.

<output_path> is the path to the directory where the tokenizer will be saved.

<langugage_list> is the list of langauges' iso codes (e.g. en es pl).


### Training LMs
Before training the LM, it's necessary to prepare training configuration files:

```commandline
source prepare_mlm_training_config.sh <vocab_size> <alpha_idx> <alpha_idx_train> <path_to_data_dir> <output_path> <langugage_list>
```

<alpha_idx_train> is choice og alpha parameter for training LM (note it can be different than <alpha_idx> for tokenizer training).

<output_path> is the path to the directory with subdirectories: /tokenizes and /models. All experimental outputs will
be saved in those subdirectories.

Other parameters are the same as above.

Now, you can train the LM with the following command, with the same parameters:

```commandline
source pretrtain_mlm.sh <vocab_size> <alpha_idx> <alpha_idx_train> <tokenizer_type> <path_to_data_dir> <output_path> <langugage_list>
```

### Fine tuning models on downstream tasks

To fine-tune the models on downstream task run (for example for NER task):

```commandline
source fine_tune_ner.sh <vocab_size> <alpha_> <alpha_train> <language_code> <tokenizer_type> <r_seed> <do_probe> <output_path>
```

New parameters are:

<alpha_> and <alpha_train> are now numerical values instead of indices.

<language_code> is the iso code language of the downstream task (e.g. en es pl).

<r_seed> is the seed for the random initialization of the probe.

<do_probe> 0 or 1: whether to use probing setting (i.e. freezeing the underlying LM's parameteres).

Subsequently, you can evaluate the fine-tuned model on test data in the same or different language.

```commandline
source eval_ner.sh <vocab_size> <alpha> <alpha_train> <language_src> <language_tgt> <tokenizer_type> <r_seed> <do_probe> <output_path>
```
where <language_src> is the language of the fine-tuned model and <language_tgt> is the language of the test data.


## Bibtex
TODO: provide bibtex key.

If you use the code, please cite the paper:
```bibtex


```