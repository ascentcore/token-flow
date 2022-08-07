from tabnanny import process_tokens
import spacy


class Vocabulary():

    listeners = []

    def __init__(self,
                 vocabulary=None,
                 include_start=True,
                 accepted=['NOUN', 'PROPN', 'ADJ', 'VERB'],
                 use_lemma=True,
                 use_token=True,
                 add_token_to_vocab=True,
                 add_lemma_to_vocab=True):
        self.nlp = spacy.load("en_core_web_sm")
        self.accepted = accepted
        self.use_lemma = use_lemma
        self.use_token = use_token
        self.add_token_to_vocab = add_token_to_vocab
        self.add_lemma_to_vocab = add_lemma_to_vocab
        self.vocabulary = vocabulary if vocabulary is not None else [
            '<start>'] if include_start else []

    @classmethod
    def from_text(cls, text, include_start = True):
        vocab = cls(include_start = include_start)
        vocab.add_text(text)

        return vocab

    @classmethod
    def from_list(cls, vocabulary):
        vocab = cls(vocabulary=vocabulary)
        return vocab

    @classmethod
    def from_file(cls, path, name = 'vocabulary.txt'):
        local = []
        with open(f'{path}/{name}', 'r') as fp:
            for line in fp:
                local.append(line[:-1])

        return cls(vocabulary=local)

    def size(self):
        return len(self.vocabulary)

    def save_vocabulary(self, path, file = "vocabulary.txt"):
        with open(f'{path}/{file}', 'w') as fp:
            fp.write("\n".join(str(item)
                     for item in self.vocabulary))

    def register(self, listener_fn):
        self.listeners.append(listener_fn)

    def add_to_vocabulary(self, text):
        if text not in self.vocabulary:
            self.vocabulary.append(text)
            return True

        return False

    def process_token(self, token, sequence, missing, append_to_vocab=True, accept_all=False):
        if accept_all or token.pos_ in self.accepted:
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

    def get_token_sequence(self, text, append_to_vocab=True, include_start=True, accept_all=True):
        doc = self.nlp(text.lower())
        missing = []
        sequences = []

        for sent in doc.sents:
            sequence = [['<start>']] if include_start else []
            for token in sent:
                self.process_token(
                    token, sequence, missing, append_to_vocab=append_to_vocab, accept_all=accept_all)

            sequences.append(sequence)

        if len(missing) > 0:
            for listener in self.listeners:
                listener(missing)

        return missing, sequences

    def add_text(self, text, include_start=True, accept_all=True):
        return self.get_token_sequence(text, append_to_vocab=True, include_start=include_start, accept_all=accept_all)
