import datasets
from datasets import load_dataset, ReadInstruction


class MLMDataset:
    # Usage: from mlm_dataset import MLMDataset
    # en_dataset = MLMDataset('cc100', 'en', 1)
    # Unfortunately, whole dataset will be downloaded from huggingface
    
    def __init__(self, dataset, language, train_share=10, validation_share=1, test_share=1):
        self.dataset = dataset
        self.language = language
        
        self.train_share = train_share
        self.train_val_share = train_share + validation_share
        self.train_val_test_share = train_share + validation_share + test_share

    @property
    def train(self):
        return load_dataset(self.dataset, lang=self.language, split=ReadInstruction('train', to=self.train_share, unit='%'))

    @property
    def validation(self):
        return load_dataset(self.dataset, lang=self.language, split=ReadInstruction('train', from_=self.train_share, to=self.train_val_share, unit='%'))

    @property
    def test(self):
        return load_dataset(self.dataset, lang=self.language, split=ReadInstruction('train', from_=self.train_val_share, to=self.train_val_test_share, unit='%'))
    
    