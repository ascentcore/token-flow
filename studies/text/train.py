import tqdm
import torch
import numpy as np
from torchdata.datapipes.iter import IterDataPipe
from torch.utils.data import DataLoader

from src.net.models.gptS import GPT
from src.net.trainer import Trainer
from src.embeddings.embeddings import get_embeddings

from config import config, text_file, contexts, size, history, next, vocabulary


def get_input(context):
    input = [[int(context.vocabulary.index_of(token)), stimulus] for token, stimulus in context.get_top_stimuli(size, history, next)]

    return input


class Text(IterDataPipe):

    def __init__(self, contexts, size):
        self.contexts = contexts
        self.size = size
        self.text_data = open(text_file).read()

    def __iter__(self):
        for context in self.contexts.values():
            context.decrease_stimulus(1)
        
        context = self.contexts['city_mouse']

        for phrase in self.text_data.splitlines():
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


def load_embeddings():
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
    datapipe = Text(contexts, size)
    dl = DataLoader(dataset=datapipe, batch_size=4,
                    num_workers=4)

    load_embeddings()

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
