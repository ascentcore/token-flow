import re
import os
import json
import shutil
from src.context.recorder import Recorder
from src.context.dataset import Dataset
from src.context.vocabulary import Vocabulary


vocabulary = Vocabulary(include_start_end=True,
                        include_punctuation=False, accept_all=True)


def get_context(name,
                vocab,
                initial_weight=0.5,
                weight_increase=0.01,
                temp_decrease=0.1,
                neuron_opening=0.95):

    context = Recorder(name, vocab,
                       initial_weight=initial_weight,
                       neuron_opening=neuron_opening,
                       weight_increase=weight_increase,
                       temp_decrease=temp_decrease)

    return context


chat_data = [
    ['Human 1', 'hello!'],
    ['Human 2', 'hi there!'],
    ['Human 1', 'how are you?'],
    ['Human 2', 'I\'m fine thanks, and you?'],
    ['Human 1', 'I am also good.'],
    ['Human 2', 'good.'],
]


dataset = Dataset(vocabulary)

dataset.add_context(get_context('Human 1', vocabulary))
dataset.add_context(get_context('Human 2', vocabulary))

dataset.get_context('Human 1').start_recording(
    'output/human1.gif', 'Human 1', consider_stimulus=False, fps=10, arrow_size=0.02)
dataset.get_context('Human 2').start_recording(
    'output/human2.gif', 'Human 2', consider_stimulus=False, fps=10, arrow_size=0.02)

for actor, text in chat_data:
    dataset.get_context(actor).add_text(text)

for actor, text in chat_data:
    dataset.get_dataset(actor).add_text(text)

    for rest in [context for context, text in chat_data if context != actor]:
        dataset.get_context(rest).stimulate_sequence(text)


for i in range(0, 10):

    for actor, _ in chat_data:
        dataset.get_context(actor).decrease_stimulus()
    