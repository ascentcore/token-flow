import re
import os
import json
from src.context.dataset import Dataset
from src.context.context import Context
from src.context.recorder import Recorder
from src.context.vocabulary import Vocabulary
from tqdm import tqdm
from src.net.ds.transformer_dataset import TransformerDataset
from src.net.models.vit import VisionTransformer

from src.net.trainer import Trainer

from src.net.models.autoencoder import AE
from src.net.models.residual import ResidualModel

vocabulary = Vocabulary(
    accept_all=True,
    include_start_end=False,
    include_punctuation=True,
    use_lemma=False,
    add_lemma_to_vocab=False)

initial_weight = 0.439
weight_increase = 0.013
temp_decrease = 0.08
neuron_opening = 0.5


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
    num_patches = 10

    model = VisionTransformer(
        embed_dim=32,
        hidden_dim=64,
        num_channels=1,
        num_heads=8,
        num_layers=6,
        num_classes=vocabulary.size(),
        patch_size=vocabulary.size(),
        num_patches=num_patches,
        dropout=0)

    trainer = Trainer(model, vocabulary)

    settings = json.loads(
        open(f'studies/various-texts/dataset/dataset.settings.json').read())

    contexts = []

    for context_name in settings['contexts']:
        if context_name != 'default':
            context = Context.from_file(
                'studies/various-texts/dataset', context_name, vocabulary)
            contexts.append(context)

    # pre = 'A Hare was making fun of the Tortoise one day for'
    pre = None

    for iter in range(0, 20):
        for context in contexts:
            print(f'\n############ {context.name} ############')
            ds = TransformerDataset(
                f'studies/various-texts/dataset/{context.name}.dataset.json', num_patches)
            trainer.train(ds, epochs=10)
            # trainer.train(
            #     context, f'studies/various-texts/dataset/{context.name}.dataset.json', 25)

            for c in contexts:
                c.decrease_stimulus(1)
                input_data = Dataset.get_sample_data(
                    c, "wolf house puff red grandmother")

                if pre != None:
                    c.stimulate_sequence(pre)
                text = trainer.get_sentence(
                    c, input_data, generate_length=100, prevent_convergence_history=5, num_patches=num_patches)
                print(f'\n\n{c.name}: [{pre}] {text}')


prepare_dataset()
train()
