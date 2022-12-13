import tqdm
import torch
import numpy as np
from torchdata.datapipes.iter import IterDataPipe
from torch.utils.data import DataLoader

from datetime import datetime

from src.net.models.gptS import GPT
from src.net.trainer import Trainer
from src.embeddings.embeddings import get_embeddings
from config import get_training_setup, size, history, next

def get_input(context):
    input = [[int(context.vocabulary.index_of(token)), stimulus] for token, stimulus in context.get_top_stimuli(size, history, next)]

    return input


class DataPipeline(IterDataPipe):

    def __init__(self, contexts, size):
        self.contexts = contexts
        self.size = size

    def __iter__(self):
        print('iter', datetime.now())
        for context in self.contexts.values():
            print('decreasing stimulus', datetime.now())
            context.decrease_stimulus(1)
            print('opening file', datetime.now())
            text = open(f'studies/text/datasets/train/{context.name}.txt').read()
            print('splitting', datetime.now())
            for phrase in text.splitlines():
                if phrase != '':
                    input = get_input(context)

                    _, sentences = context.vocabulary.get_token_sequence(
                        phrase, append_to_vocab=False, skip_eol=True)
                    for sentence in sentences:
                        for tokens in sentence:
                            for token in tokens:
                                context.stimulate(token)
                                output = np.zeros(
                                    context.vocabulary.size(), np.float32)
                                output[int(context.vocabulary.index_of(token))] = 1
                                yield input, output
                                input = get_input(context)

                    context.stimulate_sequence(phrase, skip_eol=True)


def load_embeddings(vocabulary, config):
    matrix_len = len(vocabulary.vocabulary)
    pretrained_embeddings = torch.zeros((matrix_len, config.n_embd))
    embeddings = get_embeddings()

    for i, word in enumerate(vocabulary.vocabulary):
        try: 
            pretrained_embeddings[i] = torch.tensor(embeddings[word])
        except KeyError:
            pretrained_embeddings[i] = torch.tensor(np.random.normal(scale=0.6, size=(config.n_embd, )))

    config.pretrained_embeddings = pretrained_embeddings


def train():
    contexts, vocabulary, config = get_training_setup()
    datapipe = DataPipeline(contexts, size)
    dl = DataLoader(dataset=datapipe, batch_size=4,
                    num_workers=1)

    load_embeddings(vocabulary=vocabulary, config=config)

    model = GPT(config)
    trainer = Trainer(model, vocabulary, config=config, lr=config.learning_rate)

    for epoch_idx in range(1):
        loss_all = 0
        batch_loss = 0
        # model.train(True)
        for (batch_idx, batch) in enumerate(dl):
            loss = trainer.batch_train(batch)


if __name__ == '__main__':
    train()
