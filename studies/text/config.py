import json

from src.context.context import Context
from src.context.vocabulary import Vocabulary

text_file = f'studies/text/datasets/train/city_mouse.txt'

initial_weight = 0.2
weight_increase = 0.037
temp_decrease = 0.08
neuron_opening = 0.75

n_dim = 50
size = 11
history = 5
next = 5

vocabulary = Vocabulary.from_file('studies/text/dataset')

settings = json.loads(open(f'studies/text/dataset/dataset.settings.json').read())

contexts_list = None
contexts = {}

for context_name in settings['contexts']:
    if context_name != 'default' and (contexts_list is None or context_name in contexts_list):
        context = Context.from_file('studies/text/dataset', context_name, vocabulary)
        contexts[context_name] = context
