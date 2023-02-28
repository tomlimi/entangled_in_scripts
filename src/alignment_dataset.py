import torch
from abc import abstractmethod
from datasets import Dataset, load_dataset
import transformers


class AlignmentDataset:
    
    _truncate_at = 1000
    
    def __init__(self, language_src, language_tgt, tokenizer_src, tokenizer_tgt=None,
                 max_length=128, lang_offset_src=0, lang_offset_tgt=0, evaluation=False):
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_length = max_length - 2
        
        self.language_src = language_src
        self.language_tgt = language_tgt
        
        self.lang_offset_src = lang_offset_src
        self.lang_offset_tgt = lang_offset_tgt
       
        self.evaluation = evaluation
        self.dataset = None

    @abstractmethod
    def get_src_sentences(self, examples):
        pass

    @abstractmethod
    def get_tgt_sentences(self, examples):
        pass

    def tokenize(self, examples) -> (dict, dict):
    
        inputs = transformers.BatchEncoding()
        inputs_src = self.tokenizer_src.batch_encode_plus(self.get_src_sentences(examples),
                                                          truncation=True, is_split_into_words=False,
                                                          padding=True,
                                                          max_length=self.max_length)
        inputs_tgt = self.tokenizer_tgt.batch_encode_plus(self.get_tgt_sentences(examples),
                                                          truncation=True, is_split_into_words=False,
                                                          padding=True,
                                                          max_length=self.max_length)
        # add offset if set for the language
        if self.lang_offset_src > 0:
            inputs_src['input_ids'] = [[tok_id + self.lang_offset_src if tok_id > 4 else tok_id for tok_id in ii]
                                       for ii in inputs_src['input_ids']]
        if self.lang_offset_tgt > 0:
            inputs_tgt['input_ids'] = [[tok_id + self.lang_offset_tgt if tok_id > 4 else tok_id for tok_id in ii]
                                       for ii in inputs_tgt['input_ids']]

        inputs['src_input_ids'] = inputs_src['input_ids']
        inputs['tgt_input_ids'] = inputs_tgt['input_ids']
        
        inputs['src_attention_mask'] = inputs_src['attention_mask']
        inputs['tgt_attention_mask'] = inputs_tgt['attention_mask']
        return inputs
    
    
class TatoebaAlignmentDataset(AlignmentDataset):
    
    _langs = {'ar': 'ar', 'tr': 'tr', 'zh': 'cmn', 'el': 'el', 'es': 'es', 'en': 'en', 'sw': 'swh', 'mr': 'mr',
                  'hi': 'hi', 'ur': 'ur', 'ta': 'ta', 'te': 'te', 'th': 'th', 'ru': 'ru', 'bg': 'bg', 'he': 'he',
                  'ka': 'ka', 'vi': 'vi', 'fr': 'fr', 'de': 'de', 'ko': 'ko', 'eu': 'eu', 'hu': 'hu'}

    def __init__(self, language_src, language_tgt, tokenizer_src, tokenizer_tgt=None,
                 max_length=128, lang_offset_src=0, lang_offset_tgt=0, evaluation=False):
        
        tokenizer_tgt = tokenizer_tgt if tokenizer_tgt is not None else tokenizer_src
        
        language_tgt = self._langs[language_tgt]
        language_src = self._langs[language_src]
        
        # order languages in alphabetical order
        if language_tgt < language_src:
            language_src, language_tgt = language_tgt, language_src
            tokenizer_src, tokenizer_tgt = tokenizer_tgt, tokenizer_src
            lang_offset_src, lang_offset_tgt = lang_offset_tgt, lang_offset_src
        try:
            self.dataset = load_dataset('tatoeba', f'{language_src}-{language_tgt}', split='train')
        # try to load the dataset with the opposite configuration of languages
        except FileNotFoundError:
            language_src, language_tgt = language_tgt, language_src
            tokenizer_src, tokenizer_tgt = tokenizer_tgt, tokenizer_src
            lang_offset_src, lang_offset_tgt = lang_offset_tgt, lang_offset_src
            self.dataset = load_dataset('tatoeba', f'{language_src}-{language_tgt}', split='train')
        
        super().__init__(language_src, language_tgt, tokenizer_src, tokenizer_tgt,
                         max_length, lang_offset_src, lang_offset_tgt, evaluation)

    def get_src_sentences(self, examples):
        return [e[self.language_src] for e in examples['translation']]
    
    def get_tgt_sentences(self, examples):
        return [e[self.language_tgt] for e in examples['translation']]
    
    @property
    def test(self) -> Dataset:
        if self._truncate_at > 0:
            return self.dataset.shuffle(seed=0).select(range(min(len(self.dataset),self._truncate_at))).\
                map(self.tokenize, batched=True)
        return self.dataset.map(self.tokenize, batched=True)

    @property
    def train(self):
        raise NotImplementedError

    @property
    def validation(self):
        raise NotImplementedError
    
    
class XtremeTatoebaAlignmentDataset(AlignmentDataset):

    _langs_iso2 = {'ar': 'ara', 'tr': 'tur', 'zh': 'cmn', 'el': 'ell', 'es': 'spa', 'en': 'eng', 'sw': 'swa', 'mr': 'mar',
                    'hi': 'hin', 'ur': 'urd', 'ta': 'tam', 'te': 'tel', 'th': 'tha', 'ru': 'rus', 'bg': 'bul', 'he': 'heb',
                    'ka': 'kat', 'vi': 'vie', 'fr': 'fra', 'de': 'deu', 'ko': 'kor', 'eu': 'eus', 'hu': 'hun'}
    
    def __init__(self, language_src, language_tgt, tokenizer_src, tokenizer_tgt=None,
                    max_length=128, lang_offset_src=0, lang_offset_tgt=0, evaluation=False):
            
            tokenizer_tgt = tokenizer_tgt if tokenizer_tgt is not None else tokenizer_src
            
            language_tgt = self._langs_iso2[language_tgt]
            language_src = self._langs_iso2[language_src]
            
            # order languages so English is always the target
            if language_src == 'eng':
                language_src, language_tgt = language_tgt, language_src
                tokenizer_src, tokenizer_tgt = tokenizer_tgt, tokenizer_src
                lang_offset_src, lang_offset_tgt = lang_offset_tgt, lang_offset_src
            elif language_tgt != 'eng':
                raise ValueError('Only alignments with English are supported in Xtreme!')
            
            super().__init__(language_src, language_tgt, tokenizer_src, tokenizer_tgt,
                            max_length, lang_offset_src, lang_offset_tgt, evaluation)
            self.dataset = load_dataset("xtreme", "tatoeba." + language_src, split="validation")
    
    def get_src_sentences(self, examples):
        return examples['source_sentence']

    def get_tgt_sentences(self, examples):
        return examples['target_sentence']
    
    @property
    def test(self) -> Dataset:
        return self.dataset.map(self.tokenize, batched=True)
    
    @property
    def train(self):
        raise NotImplementedError
    
    @property
    def validation(self):
        raise NotImplementedError