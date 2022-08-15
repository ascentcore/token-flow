from src.context.dataset import Dataset
import unittest


class TestDataset(unittest.TestCase):


    def test_initialization(self):
        dataset = Dataset()
        dataset.add_text("the rain in spain falls")

        data = dataset.get_dataset()
        data.add_text("the rain in spain falls.")
        
        self.assertEqual(len(data.data), 8)

    def test_append(self):
        dataset = Dataset()
        dataset.add_text("the rain in spain falls")

        data = dataset.get_dataset()
        data.add_text("the rain in spain falls.")
        data.add_text("spain in rain.")
        
        self.assertEqual(len(data.data), 13)


    # def test_initialization(self):
    #     dataset = Dataset()
    #     dataset.add_text('text_context', 'The rain in spain')
    
    # def test_multiple_contexts(self):
    #     dataset = Dataset()
    #     dataset.add_text('test_context1', 'The rain in spain. The rain is actually watter.')
    #     dataset.add_text('test_context2', 'Spain is a country in Europe')

    #     self.assertEqual(len(dataset.context_data.keys()), 2)
    #     self.assertEqual(dataset.vocabulary.size(), 13)

    # def test_get_full_Dataset(self):
    #     dataset = Dataset()
    #     dataset.add_text('text_context', 'The rain in spain')
    #     list = dataset.get_dataset()
        
    #     self.assertEqual(len(list), 5)