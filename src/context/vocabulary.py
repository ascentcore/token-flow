import os
import re
import spacy
import json


class Vocabulary():

    listeners = []

    def __init__(self,
                 vocabulary=None,
                 include_start_end=True,
                 include_punctuation=True,
                 accepted=['NOUN', 'PROPN', 'ADJ', 'VERB'],
                 accept_all=True,
                 use_lemma=True,
                 use_token=True,
                 add_token_to_vocab=True,
                 add_lemma_to_vocab=True):
        self.nlp = spacy.load("en_core_web_sm")

        prefixes = list(self.nlp.Defaults.prefixes)
        prefixes.remove('<')
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        self.nlp.tokenizer.prefix_search = prefix_regex.search

        suffixes = list(self.nlp.Defaults.suffixes)
        suffixes.remove('>')
        suffix_regex = spacy.util.compile_suffix_regex(suffixes)
        self.nlp.tokenizer.suffix_search = suffix_regex.search

        self.accepted = accepted
        self.accept_all = accept_all
        self.use_lemma = use_lemma
        self.use_token = use_token
        self.add_token_to_vocab = add_token_to_vocab
        self.add_lemma_to_vocab = add_lemma_to_vocab
        self.include_start_end = include_start_end
        self.include_punctuation = include_punctuation
        self.vocabulary = vocabulary if vocabulary is not None else [
            '<null>', '<start>', '<end>', '<eol>'] if include_start_end else []

        self.vectors = []

    @classmethod
    def from_text(cls, text, *args, **kwargs):
        vocab = cls(*args, **kwargs)
        vocab.add_text(text)

        return vocab

    @classmethod
    def from_list(cls, vocabulary):
        vocab = cls(vocabulary=vocabulary)
        return vocab

    def from_folder(self, folder):
        res = []
        for (dir_path, dir_names, file_names) in os.walk(folder):
            res.extend(file_names)

        for file_name in res:
            file = open(f'{folder}/{file_name}', 'r')
            for line in file.readlines():
                self.add_text(line)

        print(f'Added {len(res)} files to vocabulary')
        print(f'Vocabulary size: {self.size()}')

    @classmethod
    def from_file(cls, path, name='vocabulary.json'):
        with open(f'{path}/{name}', 'r') as infile:
            data = json.load(infile)

        vocabulary = cls(vocabulary=data['vocabulary'],
                         accepted=data['settings']['accepted'],
                         accept_all=data['settings']['accept_all'],
                         use_lemma=data['settings']['use_lemma'],
                         include_punctuation=data['settings']['include_punctuation'],
                         use_token=data['settings']['use_token'],
                         add_token_to_vocab=data['settings']['add_token_to_vocab'],
                         add_lemma_to_vocab=data['settings']['add_lemma_to_vocab'],
                         include_start_end=data['settings']['include_start_end'])

        print(f'Loaded vocabulary. Size: {vocabulary.size()}')

        return vocabulary

    def size(self):
        return len(self.vocabulary)

    def save_vocabulary(self, path, file="vocabulary.json"):
        save_data = {
            'settings': {
                'accepted': self.accepted,
                'use_lemma': self.use_lemma,
                'accept_all': self.accept_all,
                'include_punctuation': self.include_punctuation,
                'use_token': self.use_token,
                'add_token_to_vocab': self.add_token_to_vocab,
                'add_lemma_to_vocab': self.add_lemma_to_vocab,
                'include_start_end': self.include_start_end
            },
            'vocabulary': self.vocabulary
        }

        with open(f'{path}/{file}', 'w') as outfile:
            outfile.write(json.dumps(save_data, indent=2))

    def register(self, listener_fn):
        self.listeners.append(listener_fn)

    def add_to_vocabulary(self, text):
        if text not in self.vocabulary:
            self.vocabulary.append(text)
            return True

        return False

    def index_of(self, token):
        if token in self.vocabulary:
            return self.vocabulary.index(token)

        return 0

    def process_token(self, token, sequence, missing, append_to_vocab=True):
        if self.accept_all or token.pos_ in self.accepted:
            current = []
            lower = token.text

            if self.use_token:
                current.append(lower)
                if append_to_vocab and self.add_token_to_vocab and self.add_to_vocabulary(lower):
                    missing.append(lower)

            if self.use_lemma and (lower != token.lemma_ or not self.use_token):
                current.append(token.lemma_)
                if append_to_vocab and self.add_lemma_to_vocab and self.add_to_vocabulary(token.lemma_):
                    missing.append(token.lemma_)

            sequence.append(current)

    def get_token_sequence(self, text, append_to_vocab=True, skip_eol=False):
        text = text.lower()

        text = re.sub(r"\.([a-zA-Z0-9])", r'. \1', text)
        text = re.sub(r"([a-zA-Z0-9])\.", r'\1 .', text)
        text = re.sub(r"([0-9])", r' \1 ', text)
        text = re.sub(r"[ ]{2,}", r' ', text)
        text = re.sub(r"([a-zA-Z0-9])\,", r'\1 ,', text)
        text = re.sub(r"\?([a-zA-Z0-9])", r'? \1', text)
        text = text.strip()

        doc = self.nlp(text.lower().strip())

        missing = []
        sequences = []

        should_start = True

        for sent in doc.sents:
            sequence = []
            for token in sent:
                if token.text.strip() != '':
                    if should_start and self.include_start_end:
                        sequence.append(['<start>'])
                        should_start = False

                    if token.is_punct:
                        if self.include_start_end and token.text == '.':
                            sequence.append(['<end>'])
                            should_start = True
                        elif self.include_punctuation:
                            sequence.append([token.text])
                            if append_to_vocab and self.add_to_vocabulary(token.text):
                                missing.append(token.text)
                    else:
                        self.process_token(
                            token, sequence, missing, append_to_vocab=append_to_vocab)

            sequences.append(sequence)

        if self.include_start_end and len(sequences) > 0 and skip_eol is False:
            sequences[-1].append(['<eol>'])

        if len(missing) > 0:
            for listener in self.listeners:
                listener(missing)
        return missing, sequences

    def add_text(self, text, append_to_vocab=True):
        return self.get_token_sequence(text, append_to_vocab=append_to_vocab)
