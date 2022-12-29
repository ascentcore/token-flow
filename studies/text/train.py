import os
import torch
import numpy as np
from math import nan
from tqdm import tqdm
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
        # print('iter', datetime.now())
        for context in self.contexts.values():
            # print('decreasing stimulus', datetime.now())
            context.decrease_stimulus(1)
            # print('opening file', datetime.now())
            text = open(f'studies/text/datasets/train/{context.name}.txt').read()
            # print('splitting', datetime.now())
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
            if word == '<null>':
                pretrained_embeddings[i] = torch.tensor(np.zeros(config.n_embd))
            else:
                pretrained_embeddings[i] = torch.tensor(np.random.normal(scale=0.6, size=(config.n_embd, )))

    config.pretrained_embeddings = pretrained_embeddings


def train():
    contexts, vocabulary, config = get_training_setup()
    datapipe = DataPipeline(contexts, size)
    dl = DataLoader(dataset=datapipe, batch_size=32,
                    num_workers=1)

    model_name = f"gpt-2-{config.block_size}-{config.vocab_size}-{config.n_embd}-{config.n_layer}-{config.n_head}"
    print(model_name)

    load_embeddings(vocabulary=vocabulary, config=config)

    model = GPT(config)
    trainer = Trainer(model, vocabulary, config=config, lr=config.learning_rate)
    print('model', model)
    
    for epoch_idx in range(500):
        loss_all = 0
        batch_loss = 0
        model.train(True)
        acc_batchs = []
        for (batch_idx, batch) in tqdm(enumerate(dl)):
            loss, acc = trainer.batch_train(batch)
            loss_all = loss_all + loss
            batch_loss = batch_loss + loss
    
            if batch_idx % 10 == 0:
                acc_batchs.append((batch_idx, acc))

            if batch_idx % 100 == 0 and batch_idx > 2:
                ckpt_path = os.path.join(
                    'studies/text/models/', f'{model_name}.pt')
                torch.save({'model_state_dict': model.state_dict(),
                            'optmizer_state_dict': trainer.optimizer.state_dict()
                            }, ckpt_path)

                # print(f'Batch idx {batch_idx} loss: {loss_all}\n')
                loss_all = 0

            if loss_all == nan:
                print('Batch loss is nan, stopping training')
                break

        if batch_loss == nan:
            print('Glboal loss is nan, stopping training')
            break

        ckpt_path = os.path.join(
            'studies/text/models/', f'{model_name}-epoch-end.pt')
        torch.save({'model_state_dict': model.state_dict(),
                    'optmizer_state_dict': trainer.optimizer.state_dict()
                    }, ckpt_path)
        
        print('')
        print('---------------------------------------------------------')
        print(f'%% Epoch {epoch_idx} loss: {batch_loss}\n')
        print(f'Accuracy: {acc_batchs}')
        print('---------------------------------------------------------')

if __name__ == '__main__':
    train()
