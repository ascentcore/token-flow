from src.context.dataset import Dataset
from src.context.vocabulary import Vocabulary
from src.context.context import Context

import networkx as nx

dataset = Dataset.load('studies/various-texts/dataset')
dataset.delete_context('default')

dataset.get_context('snow-white.txt').stimulate('named')