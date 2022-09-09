import re
import os
import json
from src.context.dataset import Dataset
from src.context.context import Context
from src.context.vocabulary import Vocabulary
from src.net.ds.transformer_dataset import TransformerDataset
from src.net.models.rnntrans import RNNModel, TransformerModel
from src.net.models.vit import VisionTransformer
import torch.nn as nn

from src.net.trainer import Trainer

from src.net.models.autoencoder import AE
from src.net.models.residual import ResidualModel

vocabulary = Vocabulary(
    accept_all=True,
    include_start_end=True,
    include_punctuation=False,
    use_lemma=False,
    add_lemma_to_vocab=False)

initial_weight = 0.3
weight_increase = 0.013
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


def prepare_dataset():

    dataset = Dataset(vocabulary)
    dataset.from_folder('studies/various-texts/texts', get_context)
    dataset.store('studies/various-texts/dataset')


def train():

    vocabulary = Vocabulary.from_file(
        'studies/various-texts/dataset', 'vocabulary.json')
    # model = AE(vocabulary.size())
    # model = ResidualModel(vocabulary.size())
    num_patches = 8

    model = VisionTransformer(
        embed_dim=128,
        hidden_dim=256,
        num_channels=1,
        num_heads=4,
        num_layers=6,
        num_classes=vocabulary.size(),
        patch_size=vocabulary.size(),
        num_patches=num_patches,
        dropout=0.05)

    trainer = Trainer(model, vocabulary)

    settings = json.loads(
        open(f'studies/various-texts/dataset/dataset.settings.json').read())

    contexts = []

    for context_name in settings['contexts']:
        if context_name != 'default':
            context = Context.from_file(
                'studies/various-texts/dataset', context_name, vocabulary)
            contexts.append(context)

    pre = "the industrial revolution started with the emergence of the steam engine"

    for iter in range(0, 100):
        for context in contexts:
            print(f'\n############ {context.name} ############')
            ds = TransformerDataset(
                f'studies/various-texts/dataset/{context.name}.dataset.json', num_patches)
            trainer.train(ds, epochs=50, batch_size=32)

            for c in contexts:
                c.decrease_stimulus(1)

                input_data = Dataset.get_sample_data(c, pre)
                text = trainer.get_sentence(
                    c, input_data, generate_length=150, prevent_convergence_history=5, num_patches=num_patches)
                print(f'\n\n{c.name}: [{pre}] {text}')


prepare_dataset()
train()
