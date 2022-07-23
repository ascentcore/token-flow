from src.dataset import Dataset
import unittest


class TestDataset(unittest.TestCase):

    def test_initialization(self):
        dataset = Dataset()
        dataset.add_text('text_context', 'The rain in spain')
    
    def test_multiple_contexts(self):
        dataset = Dataset()
        dataset.add_text('test_context1', 'The rain in spain. The rain is actually watter.')
        dataset.add_text('test_context2', 'Spain is a country in Europe')

        self.assertEqual(len(dataset.contexts.keys()), 2)
        self.assertEqual(dataset.vocabulary.size(), 12)

    def test_get_full_Dataset(self):
        dataset = Dataset()
        dataset.add_text('text_context', 'The rain in spain')
        list = dataset.get_dataset()
        
        self.assertEqual(len(list), 5)