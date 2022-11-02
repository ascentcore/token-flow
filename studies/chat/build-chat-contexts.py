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


def get_context(name,
                initial_weight=config.initial_weight,
                weight_increase=config.weight_increase,
                temp_decrease=config.temp_decrease,
                neuron_opening=config.neuron_opening):

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
        open(config.chat_file).read())
    index = 0
    for id in tqdm(chat_data.keys()):
        content = chat_data[id]["content"]
        for data in content:
            agent = 'agent_1' if index % 2 == 0 else 'agent_2'
            message = data["message"]

            if not dataset.has_context(agent):
                context = get_context(agent)
                dataset.add_context(context)
            else:
                context = dataset.get_context(agent)

            context.add_text(message)
        index += 1

    print('Vocabulary size:', len(vocabulary.vocabulary))

    dataset.store('studies/chat/dataset')


if __name__ == '__main__':
    build_dataset()
