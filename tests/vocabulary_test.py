
import unittest
import os
import shutil
import pathlib as pl
import itertools

from src.context.vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):

    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def test_from_string(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.')
        print(vocab.vocabulary)
        self.assertEqual(vocab.size(), 11)

    def test_from_string_accept_pref(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.', accept_all=False)
        print(vocab.vocabulary)
        self.assertEqual(vocab.size(), 7)

    def test_strange_case(self):
        vocab = Vocabulary(accept_all=True,
                           include_start_end=True,
                           include_punctuation=True,
                           use_lemma=False,
                           add_lemma_to_vocab=False)
        missing, seq = vocab.add_text("hello I'm good!")
        print(missing)

    def test_puncts_string(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain; falls, mainly on the plain.', use_lemma=False)
        _, result = vocab.get_token_sequence(
            'Thus, the capitain, was not found; but we agreed to resume the search.')
        result = list(itertools.chain(*result[0]))

        self.assertListEqual(result, ['<start>', 'thus', ',', 'the', 'capitain', ',', 'was',
                             'not', 'found', ';', 'but', 'we', 'agreed', 'to', 'resume', 'the', 'search', '<end>'])

    def test_no_puncts_string(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain; falls, mainly on the plain.', use_lemma=False, include_punctuation=False)
        _, result = vocab.get_token_sequence(
            'Thus, the capitain, was not found; but we agreed to resume the search.')
        result = list(itertools.chain(*result[0]))

        self.assertListEqual(result, ['<start>', 'thus', 'the', 'capitain',  'was',
                             'not', 'found',  'but', 'we', 'agreed', 'to', 'resume', 'the', 'search', '<end>'])

    def test_no_puncts_no_start_end_string(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain; falls, mainly on the plain.',
            use_lemma=False,
            include_start_end=False,
            include_punctuation=False)

        _, result = vocab.get_token_sequence(
            'Thus, the capitain, was not found; but we agreed to resume the search.')

        print(result)
        result = list(itertools.chain(*result[0]))

        self.assertListEqual(result, ['thus', 'the', 'capitain',  'was',
                             'not', 'found',  'but', 'we', 'agreed', 'to', 'resume', 'the', 'search'])

    def test_from_string_no_start_end(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.', include_start_end=False)
        self.assertEqual(vocab.size(), 10)

    def test_no_lemma_vocabulary(self):
        vocab = Vocabulary(include_start_end=False, use_lemma=False)
        vocab.add_text('The rain in Spain falls mainly on the plain.')
        self.assertListEqual(vocab.vocabulary, [
                             'the', 'rain', 'in', 'spain', 'falls', 'mainly', 'on', 'plain', '.'])

    def test_no_token_vocabulary(self):
        vocab = Vocabulary(use_token=False)
        vocab.add_text('The rain in Spain falls mainly on the plain.')
        self.assertListEqual(vocab.vocabulary, [
                             '<start>', '<end>', 'the', 'rain', 'in', 'spain', 'fall', 'mainly', 'on', 'plain'])

    def test_no_token_vocabulary_no_start_end(self):
        vocab = Vocabulary(include_start_end=False, use_token=False)
        vocab.add_text('The rain in Spain falls mainly on the plain.')
        self.assertListEqual(vocab.vocabulary, [
                             'the', 'rain', 'in', 'spain', 'fall', 'mainly', 'on', 'plain', '.'])

    def test_append_to_vocab(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.')
        missing, sequences = vocab.add_text('The rain helps plants grow.')
        self.assertListEqual(
            missing, ['helps', 'help', 'plants', 'plant', 'grow'])
        self.assertListEqual(sequences[0][3], ['helps', 'help'])
        self.assertEqual(vocab.size(), 16)

    def test_sequences(self):
        vocab = Vocabulary()
        added, sequences = vocab.add_text(
            'The rain in Spain falls mainly on the plain. The rain helps plants grow.')
        self.assertEqual(len(added), 14)
        self.assertEqual(len(sequences), 2)
        self.assertEqual(len(sequences[0]), 11)
        self.assertEqual(len(sequences[1]), 7)

    def test_save_vocabulary(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.')
        vocab.add_text('The rain helps plants grow.')
        vocab.save_vocabulary('output/tests')
        path = pl.Path("output/tests/vocabulary.json")
        self.assertIsFile(path)

    def test_load_vocabulary(self):

        self.test_save_vocabulary()

        vocab = Vocabulary.from_file('output/tests', 'vocabulary.json')
        self.assertListEqual(vocab.vocabulary, ['<start>', '<end>', 'the', 'rain', 'in', 'spain',
                             'falls', 'fall', 'mainly', 'on', 'plain', 'helps', 'help', 'plants', 'plant', 'grow'])

    def test_load_vocabulary_param(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.', include_punctuation=False, include_start_end=False, use_lemma=False)
        vocab.add_text('The rain helps plants grow.')
        vocab.save_vocabulary('output/tests', 'vocabulary-2.json')

        vocab = Vocabulary.from_file('output/tests', 'vocabulary-2.json')
        vocab.add_text('The child runs fast.')
        self.assertListEqual(vocab.vocabulary, ['the', 'rain', 'in', 'spain', 'falls',
                             'mainly', 'on', 'plain', 'helps', 'plants', 'grow', 'child', 'runs', 'fast'])

    @classmethod
    def setUpClass(cls):
        os.mkdir('output/tests')

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree('output/tests')
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == '__main__':
    unittest.main()
