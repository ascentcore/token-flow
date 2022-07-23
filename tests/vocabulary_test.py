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
        self.assertEqual(vocab.size(), 10)

    def test_append_to_vocab(self):
        vocab2 = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.')
        vocab2.add_text('The rain helps plants grow.')
        self.assertEqual(vocab2.size(), 13)

    def test_sequences(self):
        vocab = Vocabulary()
        added, sequences = vocab.add_text(
            'The rain in Spain falls mainly on the plain. The rain helps plants grow.')
        self.assertEqual(len(added), 12)
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
        self.assertEqual(vocab.size(), 13)


if __name__ == '__main__':
    unittest.main()
