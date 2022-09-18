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
import torch

from src.net.trainer import Trainer

from src.net.models.autoencoder import AE
from src.net.models.residual import ResidualModel


def train():

    vocabulary = Vocabulary.from_file(
        'studies/chat/dataset', 'vocabulary.json')

    num_patches = 8

    model = VisionTransformer(
        embed_dim=32,
        hidden_dim=64,
        num_channels=1,
        num_heads=4,
        num_layers=32,
        num_classes=vocabulary.size(),
        patch_size=vocabulary.size(),
        num_patches=num_patches,
        dropout=0.02)

    trainer = Trainer(model, vocabulary)

    settings = json.loads(
        open(f'studies/chat/dataset/dataset.settings.json').read())

    contexts_list = None  # ['Barbrady']
    contexts = []

    for context_name in settings['contexts']:
        if context_name != 'default' and (contexts_list is None or context_name in contexts_list):
            context = Context.from_file(
                'studies/chat/dataset', context_name, vocabulary)
            contexts.append(context)

    stimulate_with = "What about cats, do you like cats?"
    pre = ""

    for iter in range(0, 100):
        for context in contexts:
            print(f'\n############ {context.name} ############')
            ds = TransformerDataset(
                f'studies/chat/dataset/{context.name}.dataset.json', num_patches)
            trainer.train(ds, epochs=25, batch_size=32)

            for c in contexts:
                c.decrease_stimulus(1)

                input_data = Dataset.get_sample_data(c, pre)
                c.stimulate_sequence(stimulate_with)
                text = trainer.get_sentence(
                    c,
                    input_data,
                    generate_length=100,
                    prevent_convergence_history=3,
                    num_patches=num_patches,
                    break_on_eol=True)
                print(f'\n\n[{stimulate_with}]{c.name}: <{pre}> {text}')

        try:
            os.mkdir('studies/chat/models/')
        except:
            pass

        torch.save(model, f'studies/chat/models/model_{iter}')


if __name__ == '__main__':
    train()
