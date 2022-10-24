import numpy as np
import math

from src.context.vocabulary import Vocabulary

allowed_chars = list(set('abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’\'"/\|_@#$%^&*~`+-=<>()[]{}'))
len_of_allowed_chars = len(allowed_chars)
class Embeddings(Vocabulary):

    def __init__(self, n_dim=2, total_space=len_of_allowed_chars, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_dim = n_dim
        self.total = total_space
        self.cache = {}
        self.dim_len = math.ceil(math.pow(self.total, 1 / n_dim))
        self.cache_missing(self.vocabulary)

    def getForIndex(self, index):
        dim_arr = []
        rest = index
        for i in reversed(range(0, self.n_dim)):
            dim = rest / math.pow(self.dim_len, i)
            rest = rest % math.pow(self.dim_len, i)

            dim_arr.append(math.floor(dim))

        dim_arr.reverse()
        return dim_arr

    def get_location(self, word):
        letters = list(word)
      
        letter_locations = [self.getForIndex(allowed_chars.index(letter))
                            for letter in letters]

        
        weighted_letter_locations = [
            np.array(location) / self.dim_len * 10**(-i) for i, location in enumerate(letter_locations)]
        word_location = [sum(x) for x in zip(*weighted_letter_locations)]

        return word_location

    def cache_missing(self, missing):
        for word in missing:
            if word not in self.cache.keys():
                self.cache[word] = self.get_location(word)

    def add_text(self, text, append_to_vocab=True):
        missing, sequences = super().add_text(text, append_to_vocab)
        self.cache_missing(missing)
        return missing, sequences

    def get_token_sequence(self, text, append_to_vocab=True):
        missing, sequences = super().get_token_sequence(text, append_to_vocab)
        self.cache_missing(missing)
        return missing, sequences

    def closest_from_token(self, token):
        return self.closest(self.get_location(token))

    def closest(self, token_location):
        values = self.cache.values()
        distances = [np.linalg.norm(
            np.array(token_location) - np.array(location)) for location in values]

        return list(self.cache.keys())[distances.index(min(distances))]


if __name__ == '__main__':

    embeddings = Embeddings(
        accept_all=True,
        include_start_end=True,
        include_punctuation=True,
        use_lemma=False,
        add_lemma_to_vocab=False, n_dim=2)

    embeddings.add_text('the rain in spain falls mainly on the plain.')
