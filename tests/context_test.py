import os
import shutil

from src.context.context import Context
from src.context.vocabulary import Vocabulary

import unittest


class TestContext(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            os.mkdir('output/tests')
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree('output/tests')
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    def test_initialization_lemma_included(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.')
        context = Context('testcontext', vocabulary=vocab)
        self.assertEqual(len(context.graph.nodes), 12)
        self.assertEqual(context.graph.nodes(data=True)['the']['s'], 0)

    def test_initialization_lemma_excluded(self):
        vocab = Vocabulary(use_lemma=False)
        vocab.add_text(
            'The rain in Spain falls mainly on the plain.')
        context = Context('testcontext', vocabulary=vocab)
        self.assertEqual(len(context.graph.nodes), 11)
        self.assertEqual(context.graph.nodes(data=True)['the']['s'], 0)

    def test_add_text(self):
        context = Context('testcontext', Vocabulary(), initial_weight=0.7)
        context.add_text('The rain in Spain falls mainly on the plain.')
        context.add_text('Spain is a country in Europe.')
        context.add_text(
            'Europe is a continent, also recognised as a part of Eurasia.')
        self.assertEqual(len(context.graph.nodes), 26)

    def test_add_definition_simple(self):
        context = Context('test', Vocabulary(
            include_start_end=False, use_lemma=False))
        context.add_definition(
            'tomato', 'is a berry fruits plant')
        flatted = context.get_matrix().flatten().tolist()[0]
        self.assertListEqual(flatted, [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])

    def test_stimulus(self):
        context = Context('testcontext', Vocabulary(), initial_weight=0.2)
        context.add_text('The rain in Spain falls mainly on the plain.')
        context.add_text('Spain is a country in Europe.')
        context.add_text(
            'Europe is a continent, also recognised as a part of Eurasia.')

        context.stimulate('rain', decrease_factor=0)
        context.stimulate('in')
        context.render('output/tests/context.png', consider_stimulus=True)
        self.assertEqual(context.get_stimulus_of('rain'), 0.9)
        self.assertAlmostEqual(context.get_stimulus_of('spain'), 0.18)
        self.assertEqual(context.get_stimulus_of('in'), 1)
        self.assertAlmostEqual(context.get_stimulus_of('falls'), 0.0324)

    def test_stimulus_matrix(self):
        context = Context('testcontext', Vocabulary(), initial_weight=0.2)
        context.add_text('a b c d e')
        context.add_text('a b')
        context.add_text('a b c')
        context.add_text('a b c d e')
        context.add_text('c e d')
        expected = [0., 0., 0., 0.5, 0., 0.2, 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.5, 0., 0., 0.,
                    0., 0., 0.2, 0., 0., 0.4, 0., 0.,
                    0., 0., 0.2, 0., 0.,
                    0., 0.3, 0.2,
                    0., 0., 0.2, 0., 0., 0., 0., 0.3,
                    0., 0., 0.3, 0., 0., 0., 0.2, 0.]
        result = context.get_matrix().flatten().tolist()[0]
        for i in range(len(expected)):
            self.assertAlmostEqual(expected[i], result[i])

    def test_fully_weighted_depth(self):
        context = Context('testcontext', Vocabulary(), initial_weight=1)
        context.add_text('The rain in Spain falls mainly on the plain.')
        context.add_text('Spain is a country in Europe.')
        context.add_text(
            'Europe is a continent, also recognised as a part of Eurasia.')

        context.stimulate('rain', decrease_factor=0, max_depth=10)
        context.stimulate('spain', decrease_factor=0, max_depth=10)
        # context.pretty_print(sorted=True)

        self.assertListEqual(context.get_stimuli(), [0, 0.7290000000000001, 0.6561000000000001, 0.6561000000000001, 1, 0.9, 1, 0.9, 0.9, 0.81, 0.7290000000000001, 0.5904900000000002, 0.9, 0.9, 0.81, 0.7290000000000001,
                             0.81, 0.7290000000000001, 0.6561000000000001, 0.5904900000000002, 0.5314410000000002, 0.5314410000000002, 0.47829690000000014, 0.7290000000000001, 0.6561000000000001, 0.5904900000000002])

    def test_single_vocab_multiple_contexts(self):
        vocab = Vocabulary()
        context1 = Context('context1', vocabulary=vocab, initial_weight=0.5)
        context2 = Context('context2', vocabulary=vocab, initial_weight=0.2)

        context1.add_text(
            'An engine is a computer software')
        context2.add_text(
            'The engine a machine for converting energy into motion')

        context1.stimulate('engine')
        context2.stimulate('engine')

        self.assertAlmostEqual(context1.get_stimulus_of('computer'), 0.091125)
        self.assertAlmostEqual(context2.get_stimulus_of('computer'), 0)
        self.assertAlmostEqual(context1.get_stimulus_of('machine'), 0)
        self.assertAlmostEqual(context2.get_stimulus_of('machine'), 0.0324)

        self.assertEqual(len(vocab.vocabulary), 18)

    def test_storage(self):
        text = 'The rain in Spain falls mainly on the plain.'
        vocab = Vocabulary.from_text(text)
        context = Context('testcontext', vocabulary=vocab)
        context.add_text(text)
        context.store('output/tests')

    def test_load(self):
        text = 'The rain in Spain falls mainly on the plain.'
        vocab = Vocabulary.from_text(text)
        context = Context('testcontext', vocabulary=vocab, initial_weight=0.7)
        context.add_text(text)
        context.store('output/tests')

        context2 = Context.from_file(
            'output/tests', 'testcontext', vocabulary=vocab)
        context2.stimulate('spain')
        self.assertListEqual(context2.get_stimuli(), [
                             0, 0, 0, 0.15752960999999996, 0, 0, 1, 0.63, 0.63, 0.3969, 0.25004699999999996, 0])

    def test_top_stimuli(self):
        context = Context('testcontext', Vocabulary(),  initial_weight=0.5)
        context.add_text(
            "eclipse is a time when the moon comes between the earth and the sun, hiding the sun's light.")
        context.add_text(
            "eclipse is a time when the earth comes between the sun and the moon, hiding the moon's light.")

        context.stimulate('earth')

        self.assertListEqual(context.get_top_stimuli(5), [(
            'earth', 1), ('comes', 0.45), ('come', 0.45), ('and', 0.45), ('the', 0.24300000000000002)])

    def test_matrix(self):
        context = Context('test', Vocabulary())
        context.add_text('The rain in spain rain in spain the')
        context.stimulate('rain')
        matrix = context.get_matrix()
        self.assertEqual(matrix.shape, (7, 7))
        flatted = matrix.flatten().tolist()[0]
        self.assertAlmostEqual(flatted, [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0])
        self.assertListEqual(context.get_stimuli(), [
                             0, 0, 0, 0, 1, 0.18000000000000002, 0.032400000000000005])

    def test_edges_on_add_text(self):
        context = Context('test', Vocabulary(use_lemma=False))
        context.add_text(
            'The tomato is part of fruits family.')
        self.assertListEqual(list(context.graph.edges), [('<start>', 'the'), ('<end>', '<eol>'), ('the', 'tomato'), (
            'tomato', 'is'), ('is', 'part'), ('part', 'of'), ('of', 'fruits'), ('fruits', 'family'), ('family', '<end>')])


    def test_edges_on_add_definition(self):
        context = Context('test', Vocabulary())
        context.add_definition(
            'tomato', 'The tomato is part of fruits family.')

        self.assertListEqual(list(context.graph.edges), [('<start>', 'tomato'), ('<end>', 'tomato'), ('<eol>', 'tomato'), ('the', 'tomato'), ('tomato', '<start>'), ('tomato', 'the'), ('tomato', 'is'), ('tomato', 'be'), ('tomato', 'part'), ('tomato', 'of'), (
            'tomato', 'fruits'), ('tomato', 'fruit'), ('tomato', 'family'), ('tomato', '<end>'),  ('tomato', '<eol>'), ('is', 'tomato'), ('be', 'tomato'), ('part', 'tomato'), ('of', 'tomato'), ('fruits', 'tomato'), ('fruit', 'tomato'), ('family', 'tomato')])


if __name__ == '__main__':
    unittest.main()
