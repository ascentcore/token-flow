from math import nan
import os
import torch
from tqdm import tqdm
from torchdata.datapipes.iter import IterDataPipe
import numpy as np
from torch.utils.data import DataLoader
from src.context.context import Context
from src.context.embeddings import Embeddings
from src.net.models.gpt2 import GPT

from src.net.models.residual import ResidualModel
from src.net.models.autoencoder import AE
from src.net.models.utils import CfgNode
from src.net.trainer import Trainer

path = f'studies/index_test'
n_dim = 8
size = 40
include_stimulus = False


def get_input(context):
    null_loc = context.vocabulary.get_location('<null>')
    if include_stimulus:
        input = [(context.vocabulary.get_location(token) + [stimulus] if stimulus !=
                  0 else null_loc + [0]) for token, stimulus in context.get_top_stimuli(size)]
    else:
        input = [(context.vocabulary.get_location(token) if token !=
                  '<null>' else null_loc) for token, stimulus in context.get_top_stimuli(size)]
    input = np.array(input, np.float32)

    return input


class RuntimeDP(IterDataPipe):

    def __init__(self, context, size=size):
        self.context = context
        self.size = size
        self.text = self.context.load_text_file(
            os.path.join(path, 'test.txt'), append_to_vocab=True)

    def __iter__(self):
        if include_stimulus:
            null_loc = self.context.vocabulary.get_location('<null>') + [0]
        else:
            null_loc = self.context.vocabulary.get_location('<null>')

        start_arr = []
        for i in range(0, self.size):
            start_arr.append(null_loc)
        input = np.array(start_arr, np.float32)
        # for line in self.text:
        _, sentences = self.context.vocabulary.get_token_sequence(
            self.text, append_to_vocab=False)
        for sentence in sentences:
            for tokens in sentence:
                for token in tokens:
                    self.context.stimulate(token)
                    # output = self.context.vocabulary.get_location(token)
                    output = np.zeros(self.context.vocabulary.size(), np.float32)
                    output[int(self.context.vocabulary.vocabulary.index(token))] = 1
                    yield input, output
                    input = get_input(self.context)


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
                      initial_weight=0.2,
                      weight_increase=0.08,
                      neuron_opening=0.75,
                      temp_decrease=0.037)
    datapipe = RuntimeDP(context, size)
    datapipe = datapipe.map(row_processer)
    dl = DataLoader(dataset=datapipe, batch_size=16,
                    num_workers=1, shuffle=True)
    print('###################################')
    print(f'Vocabulary size: {len(vocabulary.vocabulary)}')
    model_name = "model-with-null-gpt-2-test"
    try:
        model = torch.load(f'studies/index_test/{model_name}.pt')
    except:
        print('No model found, creating new one')
        # model = AE(size, input_channels=n_dim + 1 if include_stimulus else n_dim,
        #            output_channels=n_dim, steps=16)
        # model = ResidualModel(size, 4, n_dim)
        # model = torch.nn.Transformer()
        config = CfgNode()
        config.model_type = None
        config.vocab_size = len(vocabulary.vocabulary)
        config.block_size = size
        config.n_embd = n_dim
        config.n_layer = 10
        config.n_head = 4
        config.embd_pdrop = 0
        config.attn_pdrop = 0
        config.resid_pdrop = 0

        model = GPT(config)
    trainer = Trainer(model, vocabulary, lr=0.001)

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

        sentence = ''
        # sentence = 'the country mouse lives in a cozy nest at the bottom '
        for token in sentence.split(' '):
            context.stimulate(token)

        input = get_input(context)

        last_token = None
        
        for i in range(150):
            # vector = trainer.predict([input]).detach().numpy()
            # token = context.vocabulary.closest(vector[0])
            logits, loss = trainer.predict([input])
            result = logits[:, -1, :]
            result = result.detach().numpy()[0]
            token = context.vocabulary.vocabulary[result.argmax()]
            if (token != last_token):
                sentence = sentence + token + ' '
                context.stimulate(token)
            else:
                context.decrease_stimulus()

            input = get_input(context)

        print(sentence)

        # torch.save(model, f'studies/index_test/{model_name}.pt')


if __name__ == '__main__':
    train()
