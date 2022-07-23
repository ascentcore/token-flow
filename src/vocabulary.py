import spacy

class Vocabulary():

    listeners = []

    def __init__(self, vocabulary=['<start>']):
        self.nlp = spacy.load("en_core_web_sm")
        self.vocabulary = vocabulary

    @staticmethod
    def from_text(text):
        vocab = Vocabulary()
        vocab.add_text(text)

        return vocab

    @staticmethod
    def from_list(vocabulary):
        vocab = Vocabulary()
        vocab.vocabulary = vocabulary
        return vocab

    @staticmethod
    def from_file(path):
        local = []
        with open(f'{path}/vocabulary.txt', 'r') as fp:
            for line in fp:
                local.append(line[:-1])

        return Vocabulary(vocabulary=local)

    def size(self):
        return len(self.vocabulary)

    def save_vocabulary(self, path):
        with open(f'{path}/vocabulary.txt', 'w') as fp:
            fp.write("\n".join(str(item)
                     for item in self.vocabulary))

    def register(self, listener_fn):
        self.listeners.append(listener_fn)

    def add_text(self, text):
        doc = self.nlp(text)
        missing = []
        sequences = []

        for sent in doc.sents:
            sequence = ['<start>']
            for token in sent:
                lower = token.text.lower()
                sequence.append(lower)
                if lower not in self.vocabulary:
                    self.vocabulary.append(lower)
                    missing.append(lower)
            sequences.append(sequence)

        for listener in self.listeners:
            listener(missing)

        return missing, sequences

        # print(self.vocabulary)
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
