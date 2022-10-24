
import unittest

from src.context.embeddings import Embeddings


class TestEmbedding(unittest.TestCase):

    def test_get_location(self):
        embeddings = Embeddings.from_text(
            'The rain in Spain falls mainly on the plain.', accept_all=True,
            include_start_end=True,
            include_punctuation=False,
            use_lemma=False,
            add_lemma_to_vocab=False, n_dim=2)

        embeddings.get_token_sequence(
            'the rain in spain falls also in the mountains')

        # self.assertEqual(embeddings.closest([0, 0]), 'also')
        # self.assertEqual(embeddings.closest([1, 0]), 'mainly')
        # self.assertEqual(embeddings.closest([1, 1]), 'the')

        # self.assertEqual(embeddings.closest_from_token('spai'), 'spain')
        # self.assertEqual(embeddings.closest_from_token('fa'), 'fall')
        # self.assertEqual(embeddings.closest_from_token('th'), 'the')
        # self.assertEqual(embeddings.closest_from_token('mount'), 'mountain')

        print(embeddings.cache)

        print(embeddings.get_location('a'))
        print(embeddings.get_location('<start>'))
        print(embeddings.get_location('mou'))
        print(embeddings.get_location('moun'))
        print(embeddings.get_location('mount'))
        print(embeddings.get_location('mounta'))
        print(embeddings.get_location('mountai'))
        print(embeddings.get_location('mountain'))


if __name__ == '__main__':
    unittest.main()
