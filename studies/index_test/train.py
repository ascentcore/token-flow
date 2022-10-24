import json
from math import nan
import torch
import os
from tqdm import tqdm
from torchdata.datapipes.iter import IterDataPipe
import numpy as np
from torch.utils.data import DataLoader
from src.context.context import Context
from src.context.embeddings import Embeddings

from src.context.vocabulary import Vocabulary

from src.net.models.residual import ResidualModel
from src.net.models.autoencoder import AE
from src.net.models.rnntrans import TransformerModel
from src.net.models.vit import VisionTransformer
from src.net.trainer import Trainer

path = f'studies/index_test'
n_dim = 3
size = 40


class RuntimeDP(IterDataPipe):

    def __init__(self, context, size=size):
        self.context = context
        self.size = size
        self.text = self.context.load_text_file(
            os.path.join(path, 'test.txt'), append_to_vocab=True)

    def __iter__(self):
        input = np.zeros((self.size, n_dim), dtype=np.float32)
        # for line in self.text:
        _, sentences = self.context.vocabulary.get_token_sequence(
            self.text, append_to_vocab=False)
        for sentence in sentences:
            for tokens in sentence:
                for token in tokens:
                    self.context.stimulate(token)
                    output = self.context.vocabulary.get_location(token)                       
                    yield input, output
                    input = np.array([self.context.vocabulary.get_location(
                        token[0]) for token in self.context.get_top_stimuli(self.size)], np.float32)


def row_processer(row):
    return [np.array(row[0], np.float32), np.array(row[1], np.float32)]


def train():

    vocabulary = Embeddings(
        n_dim=n_dim,
        accept_all=True,
        include_start_end=True,
        include_punctuation=False,
        use_lemma=False,
        add_lemma_to_vocab=False)

    context = Context('test', vocabulary,
                      initial_weight=0.5,
                      weight_increase=0.08,
                      neuron_opening=0.75,
                      temp_decrease=0.037)
    datapipe = RuntimeDP(context, size)
    datapipe = datapipe.map(row_processer)
    dl = DataLoader(dataset=datapipe, batch_size=16, num_workers=1)
    print('###################################')
    print(f'Vocabulary size: {len(vocabulary.vocabulary)}')
    # model = AE(size, n_dim)
    model = ResidualModel(size, 4, n_dim)
    trainer = Trainer(model, vocabulary, lr=0.0008)


    for i in range(1000):
        loss_all = 0
        model.train(True)
        for (batch_idx, batch) in tqdm(enumerate(dl)):
            loss = trainer.batch_train(batch)
            loss_all = loss_all + loss

        if loss_all == nan:
            print('Loss is nan, stopping training')
            break

        print(f'\n\nEpoch {i} loss: {loss_all}\n')
        context.decrease_stimulus(1)
        input = np.zeros((1, size, n_dim), dtype=np.float32)

        sentence = ''
        for i in range(150):
            vector = trainer.predict(input).detach().numpy()
            token = context.vocabulary.closest(vector[0])
            sentence = sentence + token + ' '
            context.stimulate(token)
            input = np.array([[context.vocabulary.get_location(
                        token[0]) for token in context.get_top_stimuli(size)]], np.float32)
        print(sentence)


if __name__ == '__main__':
    train()
