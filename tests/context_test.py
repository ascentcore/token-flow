from src.context import Context
from src.vocabulary import Vocabulary

import unittest


class TestContext(unittest.TestCase):

    def test_initialization(self):
        vocab = Vocabulary.from_text(
            'The rain in Spain falls mainly on the plain.')
        context = Context('testcontext', vocabulary=vocab)
        self.assertEqual(len(context.graph.nodes), 10)
        self.assertEqual(context.graph.nodes(data=True)['the']['s'], 0)

    def test_add_text(self):
        context = Context('testcontext', initial_weight=0.7)
        context.add_text('The rain in Spain falls mainly on the plain.')
        context.add_text('Spain is a country in Europe.')
        context.add_text(
            'Europe is a continent, also recognised as a part of Eurasia.')

        self.assertEqual(len(context.graph.nodes), 22)
        context.render('output/tests/context.png', consider_stimulus=False)

    def test_stimulus(self):
        context = Context('testcontext', initial_weight=0.2)
        context.add_text('The rain in Spain falls mainly on the plain.')
        context.add_text('Spain is a country in Europe.')
        context.add_text(
            'Europe is a continent, also recognised as a part of Eurasia.')

        context.stimulate('rain', decrease_factor=0)
        context.stimulate('in')
        context.render('output/tests/context.png', consider_stimulus=True)
        self.assertEqual(context.get_stimulus_of('rain'), 0.8)
        self.assertAlmostEqual(context.get_stimulus_of('spain'), 0.18)
        self.assertEqual(context.get_stimulus_of('in'), 1)
        self.assertAlmostEqual(context.get_stimulus_of('falls'), 0.0324)

    def test_fully_weighted(self):
        context = Context('testcontext', initial_weight=1)
        context.add_text('The rain in Spain falls mainly on the plain.')
        context.add_text('Spain is a country in Europe.')
        context.add_text(
            'Europe is a continent, also recognised as a part of Eurasia.')

        context.stimulate('rain', decrease_factor=0)

        expected = [('<start>', 0), ('the', 0.5314410000000002), ('rain', 1), ('in', 0.9), ('spain', 0.81), ('falls', 0.7290000000000001), ('mainly', 0.6561000000000001), ('on', 0.5904900000000002), ('plain', 0.47829690000000014), ('.', 0.43046721000000016), ('is', 0.7290000000000001), ('a', 0.7217100000000002),
                    ('country', 0.6495390000000002), ('europe', 0.81), ('continent', 0.6495390000000002), (',', 0.5845851000000002), ('also', 0.5261265900000002), ('recognised', 0.47351393100000017), ('as', 0.42616253790000014), ('part', 0.6495390000000002), ('of', 0.5845851000000002), ('eurasia', 0.5261265900000002)]

        for token, s in expected:
            self.assertEqual(context.get_stimulus_of(token), s)

    def test_single_vocab_multiple_contexts(self):
        vocab = Vocabulary()
        context1 = Context('context1', vocabulary=vocab, initial_weight=0.5)
        context2 = Context('context2', vocabulary=vocab, initial_weight=0.5)

        context1.add_text(
            'An engine is a computer software')
        context2.add_text(
            'The engine a machine for converting energy into motion')

        context1.stimulate('engine')
        context2.stimulate('engine')

        context1.render('output/tests/context1.png',
                        consider_stimulus=True, force_text_rendering=True)
        context2.render('output/tests/context2.png',
                        consider_stimulus=True, force_text_rendering=True)

        self.assertEqual(len(vocab.vocabulary), 14)

    def test_storage(self):
        text = 'The rain in Spain falls mainly on the plain.'
        vocab = Vocabulary.from_text(text)
        context = Context('testcontext', vocabulary=vocab)
        context.add_text(text)
        context.store('output/tests')

    def test_matrix(self):
        context = Context('test')
        context.add_text('The rain in spain rain in spain the')
        context.stimulate('rain')
        matrix = context.get_matrix()
        self.assertEqual(matrix.shape, (5, 5))
        flatted = matrix.flatten().tolist()[0]
        self.assertListEqual(flatted, [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.1, 0.0, 0.0])
        self.assertListEqual(context.get_stimuli(), [0, 0.0029160000000000006, 1, 0.18000000000000002, 0.032400000000000005])
        stimuli = context.get_stimuli()

if __name__ == '__main__':
    unittest.main()
