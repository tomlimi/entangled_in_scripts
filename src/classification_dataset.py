from datasets import load_dataset


class ClassificationDataset:

    TAG_FIELD = ""
    NUM_LABELS = 0

    def __init__(self, language, tokenizer, truncate_at=-1, max_length=128):
        self.tokenizer = tokenizer
        # TODO: hack with too large input length
        self.max_length = max_length - 2
        self.truncate_at=truncate_at
        self.dataset = None

    def tokenize_and_align_labels(self, examples):

        inputs = self.tokenizer.batch_encode_plus(examples["tokens"], truncation=True,
                                                       is_split_into_words=True, max_length=self.max_length)
        labels = []
        for i, (label, sent) in enumerate(zip(examples[self.TAG_FIELD], examples["tokens"])):
            label_ids = [-100]
            for word_idx, word in enumerate(sent):  # Set the special tokens to -100.
                subwords = self.tokenizer.encode(word, add_special_tokens=False)

                for subword_idx, subword in enumerate(subwords):
                    if subword_idx > 0 or subword == self.tokenizer.unk_token_id:
                        label_ids.append(-100)
                    else:
                        label_ids.append(label[word_idx])
            label_ids.append(-100)
            label_ids = label_ids[:self.max_length]  # truncating labels

            assert len(inputs['input_ids'][i]) == len(label_ids)
            labels.append(label_ids)

        inputs["labels"] = labels
        return inputs

    @property
    def train(self):
        if self.truncate_at > 0:
            return self.dataset['train'][:self.truncate_at].map(self.tokenize_and_align_labels, batched=True)
        return self.dataset['train'].map(self.tokenize_and_align_labels, batched=True)

    @property
    def validation(self):
        if self.truncate_at > 0:
            return self.dataset['validation'][:self.truncate_at].map(self.tokenize_and_align_labels, batched=True)
        return self.dataset['validation'].map(self.tokenize_and_align_labels, batched=True)

    @property
    def test(self):
        if self.truncate_at > 0:
            return self.dataset['test'][:self.truncate_at].map(self.tokenize_and_align_labels, batched=True)
        return self.dataset['test'].map(self.tokenize_and_align_labels, batched=True)


class XtremePOSClassificationDataset(ClassificationDataset):

    TAG_FIELD = "pos_tags"
    NUM_LABELS = 17

    _iso2lang = {'ur': 'Urdu', 'te': 'Telugu', 'hi': 'Hindi', 'el': 'Greek', 'ko': 'Korean', 'ru': 'Russian',
                'de': 'German', 'eu': 'Basque', 'tr': 'Turkish', 'vi': 'Vietnamese', 'hu': 'Hungarian',
                'es': 'Spanish', 'en': 'English', 'ar': 'Arabic', 'zh': 'Chinese'}

    def __init__(self, language, tokenizer, truncate_at= 1, max_length=128):
        super().__init__(language, tokenizer, truncate_at, max_length)
        if language not in self._iso2lang:
            raise ValueError(f"Language {language} not supported by Xtreme POS. Pick one of: {list(self._iso2lang.keys())}")

        self.dataset = load_dataset("xtreme", "udpos." + self._iso2lang[language])


class XtremeNERClassificationDataset(ClassificationDataset):

    TAG_FIELD = "ner_tags"
    NUM_LABELS = 7

    _langs = {'ur', 'te', 'hi', 'el', 'ko', 'ru', 'de', 'eu', 'tr', 'vi', 'hu', 'es', 'en', 'ar', 'zh'}

    def __init__(self, language, tokenizer, truncate_at= 1, max_length=128):
        super().__init__(language, tokenizer, truncate_at, max_length)
        if language not in self._langs:
            raise ValueError(f"Language {language} not supported by Xtreme PANX. Pick one of: {list(self._langs)}")

        self.dataset = load_dataset("xtreme", "PAN-X." + language)

