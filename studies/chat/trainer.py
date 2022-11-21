from math import nan
import os
import json
import torch
from tqdm import tqdm
from torchdata.datapipes.iter import IterDataPipe
import numpy as np
from torch.utils.data import DataLoader
from src.context.context import Context
from src.context.embeddings import Embeddings
from src.context.vocabulary import Vocabulary
from src.net.models.gpt2 import GPT

from src.net.models.residual import ResidualModel
from src.net.models.autoencoder import AE
from src.net.models.utils import CfgNode
from src.net.trainer import Trainer
from src.embeddings.embeddings import get_embeddings
from config import n_dim, size, history, next, vocabulary, contexts, config, chat_file
path = f'studies/index_test'
include_stimulus = False


def get_input(context):
    input = [int(context.vocabulary.index_of(token)) for token, stimulus in context.get_top_stimuli(size, history, next)]
    input = np.array(input, np.int64)

    return input


class ChatDP(IterDataPipe):

    def __init__(self, contexts, size):
        self.contexts = contexts
        self.size = size
        self.chat_data = json.loads(open(chat_file).read())

    def __iter__(self):
        index = 0
        for id in self.chat_data.keys():
            # print('Switching to new content...')
            content = self.chat_data[id]["content"]
            for context in self.contexts.values():
                # print(f'Destimulating context {context.name}')
                context.decrease_stimulus(1)

            listener = self.contexts['agent_1' if index %
                                     2 == 0 else 'agent_2']
            talker = self.contexts['agent_2' if index % 2 == 0 else 'agent_1']

            for data in content:
                message = data["message"]
                input = get_input(talker)

                _, sentences = talker.vocabulary.get_token_sequence(
                    message, append_to_vocab=False, skip_eol=True)
                for sentence in sentences:
                    for tokens in sentence:
                        for token in tokens:
                            talker.stimulate(token)
                            output = np.zeros(
                                talker.vocabulary.size(), np.float32)
                            output[int(talker.vocabulary.index_of(token))] = 1
                            yield input, output
                            input = get_input(talker)

                listener.stimulate_sequence(message, skip_eol=True)
                buf = talker
                talker = listener
                listener = buf

            index += 1


def row_processer(row):
    return [np.array(row[0], np.int64), np.array(row[1], np.float32)]


def train():

    datapipe = ChatDP(contexts, size)
    datapipe = datapipe.map(row_processer)
    dl = DataLoader(dataset=datapipe, batch_size=4,
                    num_workers=4)
    print('###################################')
    print(f'Vocabulary size: {len(vocabulary.vocabulary)}')
    model_name = f"gpt-2-{config.block_size}-{config.vocab_size}-{config.n_embd}-{config.n_layer}-{config.n_head}"
    print(model_name)

    matrix_len = len(vocabulary.vocabulary)
    pretrained_embeddings = torch.zeros((matrix_len, n_dim))
    embeddings = get_embeddings()

    for i, word in enumerate(vocabulary.vocabulary):
        try: 
            pretrained_embeddings[i] = torch.tensor(embeddings[word])
        except KeyError:
            pretrained_embeddings[i] = torch.tensor(np.random.normal(scale=0.6, size=(n_dim, )))

    config.pretrained_embeddings = pretrained_embeddings

    model = GPT(config)
    trainer = Trainer(model, vocabulary,  config=config,
                      lr=config.learning_rate)
    try:
        checkpoint = torch.load(
            f'studies/chat/models/{model_name}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optmizer_state_dict'])
        model.eval()
    except:
        print('No state saved')
        pass

    for i in range(1000):
        loss_all = 0
        batch_loss = 0
        model.train(True)
        for (batch_idx, batch) in tqdm(enumerate(dl)):
            loss = trainer.batch_train(batch)
            loss_all = loss_all + loss
            batch_loss = batch_loss + loss

            if batch_idx % 100 == 0 and batch_idx > 2:
                ckpt_path = os.path.join(
                    'studies/chat/models/', f'{model_name}.pt')
                torch.save({'model_state_dict': model.state_dict(),
                            'optmizer_state_dict': trainer.optimizer.state_dict()
                            }, ckpt_path)

                print(f'Batch idx {batch_idx} loss: {loss_all}\n')
                loss_all = 0

            if loss_all == nan:
                print('Batch loss is nan, stopping training')
                break

        if batch_loss == nan:
            print('Glboal loss is nan, stopping training')
            break

        ckpt_path = os.path.join(
            'studies/chat/models/', f'{model_name}-epoch-end.pt')
        torch.save({'model_state_dict': model.state_dict(),
                    'optmizer_state_dict': trainer.optimizer.state_dict()
                    }, ckpt_path)

        print(f'%% Epoch {i} loss: {batch_loss}\n')


if __name__ == '__main__':
    train()
