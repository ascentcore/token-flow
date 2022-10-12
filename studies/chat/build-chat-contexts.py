import os
import re
import json

from tqdm import tqdm
from src.context.dataset import BasicInMemDataset, Dataset
from src.context.context import Context
from src.context.vocabulary import Vocabulary
import config as config

vocabulary = Vocabulary(
    accept_all=True,
    include_start_end=True,
    include_punctuation=True,
    use_lemma=False,
    add_lemma_to_vocab=False)

initial_weight = 0.2
weight_increase = 0.037
temp_decrease = 0.08
neuron_opening = 0.75


def get_context(name,
                initial_weight=initial_weight,
                weight_increase=weight_increase,
                temp_decrease=temp_decrease,
                neuron_opening=neuron_opening):

    context = Context(name, vocabulary,
                      initial_weight=initial_weight,
                      neuron_opening=neuron_opening,
                      weight_increase=weight_increase,
                      temp_decrease=temp_decrease)

    return context


def build_dataset():
    dataset = Dataset(vocabulary)
    dataset.delete_context('default')
    
    chat_data = json.loads(
            open(config.file).read())

    for id in tqdm(chat_data.keys()):
        content = chat_data[id]["content"]
        for data in content:
            agent = data["agent"]
            message = data["message"]

            if not dataset.has_context(agent):
                context = get_context(agent)
                dataset.add_context(context)
            else:
                context = dataset.get_context(agent)
            
            context.add_text(message)

    print('Vocabulary size:', len(vocabulary.vocabulary))


    dataset.store('studies/chat/dataset')

   

if __name__ == '__main__':
    build_dataset()
