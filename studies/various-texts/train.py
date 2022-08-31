import re
import os
import json
from src.context.dataset import Dataset
from src.context.context import Context
from src.context.recorder import Recorder
from src.context.vocabulary import Vocabulary
from tqdm import tqdm

from src.net.trainer import Trainer

from src.net.models.autoencoder import AE
from src.net.models.residual import ResidualModel

vocabulary = Vocabulary(
    accept_all=True,
    include_start_end=False,
    include_punctuation=True,
    use_lemma=False,
    add_lemma_to_vocab=False)


def get_context(name,
                initial_weight=0.6,
                weight_increase=0.05,
                temp_decrease=0.3,
                neuron_opening=0.85):

    context = Context(name, vocabulary,
                      initial_weight=initial_weight,
                      neuron_opening=neuron_opening,
                      weight_increase=weight_increase,
                      temp_decrease=temp_decrease)

    return context


def prepare_dataset():

    dataset = Dataset(vocabulary)
    dataset.from_folder('studies/various-texts/texts', get_context)
    dataset.store('studies/various-texts/dataset')


def train():

    vocabulary = Vocabulary.from_file(
        'studies/various-texts/dataset', 'vocabulary.json')
    # model = AE(vocabulary.size())
    model = ResidualModel(vocabulary.size())

    trainer = Trainer(model, vocabulary)

    settings = json.loads(
        open(f'studies/various-texts/dataset/dataset.settings.json').read())

    contexts = []

    for context_name in settings['contexts']:
        if context_name != 'default':
            context = Context.from_file(
                'studies/various-texts/dataset', context_name, vocabulary)
            contexts.append(context)

    pre = 'once upon a time'

    for iter in range(0, 500):
        for context in contexts:
            print(f'\n############ {context.name} ############')
            trainer.train(
                context, f'studies/various-texts/dataset/{context.name}.dataset.json', 25)

            for c in contexts:
                c.decrease_stimulus(1)
                if pre != None:
                    c.stimulate_sequence(pre)
                text = trainer.get_sentence(
                    c, generate_length=100, prevent_convergence_history=5)
                print(f'\n\n{c.name}: [{pre}] {text}')


prepare_dataset()
train()
