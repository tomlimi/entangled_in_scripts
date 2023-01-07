import torch
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

    def tokenize(self, examples) -> (dict, dict):
    
        inputs = transformers.BatchEncoding()
        inputs_src = self.tokenizer_src.batch_encode_plus([e[self.language_src] for e in examples['translation']],
                                                          truncation=True, is_split_into_words=False,
                                                          padding=True,
                                                          max_length=self.max_length)
        inputs_tgt = self.tokenizer_tgt.batch_encode_plus([e[self.language_tgt] for e in examples['translation']],
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

        inputs['src_input_ids'] = torch.LongTensor(inputs_src['input_ids'])
        inputs['tgt_input_ids'] = torch.LongTensor(inputs_tgt['input_ids'])
        
        inputs['src_attention_mask'] = torch.FloatTensor(inputs_src['attention_mask'])
        inputs['tgt_attention_mask'] = torch.FloatTensor(inputs_tgt['attention_mask'])
        return inputs
    
    @property
    def test(self) -> Dataset:
        if self._truncate_at > 0:
            return self.dataset['train'].shuffle(seed=0).select(range(self._truncate_at)).map(self.tokenize, batched=True)
        return self.dataset['train'].map(self.tokenize, batched=True)
    
    @property
    def train(self):
        raise NotImplementedError
    
    @property
    def validation(self):
        raise NotImplementedError
    
    
class TatoebaAlignmentDataset(AlignmentDataset):

    _langs = {'ar', 'tr', 'zh', 'el', 'es', 'en', 'sw', 'mr', 'hi', 'ur', 'ta', 'te', 'th', 'ru', 'bg', 'he', 'ka',
              'vi', 'fr', 'de', 'ko', 'eu', 'hu'}

    def __init__(self, language_src, language_tgt, tokenizer_src, tokenizer_tgt=None,
                 max_length=128, lang_offset_src=0, lang_offset_tgt=0, evaluation=False):
        
        tokenizer_tgt = tokenizer_tgt if tokenizer_tgt is not None else tokenizer_src
        
        # order languages in alphabetical order
        if language_tgt < language_src:
            language_src, language_tgt = language_tgt, language_src
            tokenizer_src, tokenizer_tgt = tokenizer_tgt, tokenizer_src
            lang_offset_src, lang_offset_tgt = lang_offset_tgt, lang_offset_src
        
        super().__init__(language_src, language_tgt, tokenizer_src, tokenizer_tgt,
                         max_length, lang_offset_src, lang_offset_tgt, evaluation)
        self.dataset = load_dataset("tatoeba", lang1=language_src, lang2=language_tgt)

        