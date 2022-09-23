from src.context.context import Context
from src.context.dataset import Dataset
from src.context.vocabulary import Vocabulary

import unittest

class TestDataset(unittest.TestCase):

    def test_initialization(self):
        dataset = Dataset()
        dataset.add_text("the rain in spain falls")

        data = dataset.get_dataset()
        data.add_text("the rain in spain falls.")

        self.assertEqual(len(data.data), 9)

    def test_append(self):
        dataset = Dataset()
        dataset.add_text("the rain in spain falls")

        data = dataset.get_dataset()
        data.add_text("the rain in spain falls.")
        data.add_text("spain in rain.")

        self.assertEqual(len(data.data), 15)

    def test_append_with_decrease_on_end(self):
        vocab = Vocabulary()
        context = Context(name="test", vocabulary=vocab,
                          initial_weight=1, temp_decrease=0)
        dataset = Dataset(vocabulary=vocab, default_context=context)
        data = dataset.get_dataset()
        data.add_text("Test sentence. Decrease test", decrease_on_end=0.25)
        data.pretty_print()
