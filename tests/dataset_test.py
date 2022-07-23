from src.dataset import Dataset
import unittest


class TestDataset(unittest.TestCase):

    def test_initialization(self):
        dataset = Dataset()
        dataset.add_text('text_context', 'The rain in spain')
        