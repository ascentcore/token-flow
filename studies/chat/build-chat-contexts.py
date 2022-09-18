import os
import re
import json

from tqdm import tqdm
from src.context.dataset import Dataset
from src.context.context import Context
from src.context.vocabulary import Vocabulary


vocabulary = Vocabulary(
    accept_all=True,
    include_start_end=True,
    include_punctuation=False,
    use_lemma=False,
    add_lemma_to_vocab=False)

initial_weight = 0.5
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
            open(f'studies/chat/datasets/alexa/train.json').read())

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
            
            # print(f'adding to {agent} message: {message}')
            context.add_text(message)

    for id in tqdm(chat_data.keys()):
        content = chat_data[id]["content"]
        for data in content:
            agent = data["agent"]
            message = data["message"]
            for context in dataset.contexts.keys():
                if context != agent:
                    dataset.get_context(context).stimulate_sequence(message)
                else:
                    dataset.get_dataset(context, context).add_text(message)
                
        dataset.reset_stimulus()

    dataset.store('studies/chat/dataset')      
            


if __name__ == '__main__':
    build_dataset()
