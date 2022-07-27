from src.vocabulary import Vocabulary
import unittest
import pathlib as pl


class TestVocabulary(unittest.TestCase):

    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def test_from_string(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.')
        self.assertEqual(vocab.size(), 11)

        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.', include_start=False)
        print(vocab.vocabulary)
        self.assertEqual(vocab.size(), 10)

    def test_no_lemma_vocabulary(self):
        vocab = Vocabulary(include_start = False, use_lemma = False)
        vocab.add_text('The rain in Spain falls mainly on the plain.')
        self.assertListEqual(vocab.vocabulary, ['the', 'rain', 'in', 'spain', 'falls', 'mainly', 'on', 'plain', '.'])

    def test_no_token_vocabulary(self):
        vocab = Vocabulary(include_start = False, use_token= False)
        vocab.add_text('The rain in Spain falls mainly on the plain.')
        self.assertListEqual(vocab.vocabulary, ['the', 'rain', 'in', 'spain', 'fall', 'mainly', 'on', 'plain', '.'])

    def test_append_to_vocab(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.')
        missing, sequences = vocab.add_text('The rain helps plants grow.')
        self.assertListEqual(missing, ['helps', 'help', 'plants', 'plant', 'grow'])
        self.assertListEqual(sequences[0][3], ['helps', 'help'])
        self.assertEqual(vocab.size(), 16)

    def test_sequences(self):
        vocab = Vocabulary()
        added, sequences = vocab.add_text(
            'The rain in Spain falls mainly on the plain. The rain helps plants grow.')
        self.assertEqual(len(added), 15)
        self.assertEqual(len(sequences), 2)
        self.assertEqual(len(sequences[0]), 11)
        self.assertEqual(len(sequences[1]), 7)


    def test_save_vocabulary(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.')
        vocab.add_text('The rain helps plants grow.')
        vocab.save_vocabulary('output/tests')

        path = pl.Path("output/tests/vocabulary.txt")
        self.assertIsFile(path)

    def test_load_vocabulary(self):
        vocab = Vocabulary.from_file('output/tests')
        self.assertEqual(vocab.size(), 16)


if __name__ == '__main__':
    unittest.main()
