import os
import torch
import random
import numpy as np
from math import nan
from tqdm import tqdm
from torchdata.datapipes.iter import IterDataPipe
from torch.utils.data import DataLoader

from datetime import datetime

from src.net.models.gptS import GPT
from src.net.trainer import Trainer
from src.embeddings.embeddings import get_embeddings
from config import get_training_setup, size, history, next, get_model_name
import config as cfg

def get_input(context):
    input = [[int(context.vocabulary.index_of(token)), stimulus] for token, stimulus in context.get_top_stimuli(size, history, next)]
    return input


class DataPipeline(IterDataPipe):

    def __init__(self, contexts, size):
        self.contexts = contexts
        self.size = size
        self.files = os.listdir('studies/single-context/datasets/train_adap')
        self.dataset = []

    def __iter__(self):
        if len(self.dataset) == 0:
            context = self.contexts['default']
            random.shuffle(self.files)
            print('self.files', self.files)
            for file in self.files:
                context.history = []
                context.decrease_stimulus(1)
                text = open(f'studies/single-context/datasets/train_adap/{file}').read()
                for phrase in text.splitlines():
                    if phrase != '':
                        input = get_input(context)

                        _, sentences = context.vocabulary.get_token_sequence(
                            phrase, append_to_vocab=False, skip_eol=True, use_lemma_if_present = True)
                        for sentence in sentences:
                            for tokens in sentence:
                                for token in tokens:
                                    context.stimulate(token)
                                    output = np.zeros(
                                        context.vocabulary.size(), np.float32)
                                    output[int(context.vocabulary.index_of(token))] = 1
                                    self.dataset.append( (input, output) )
                                    yield input, output
                                    input = get_input(context)

            random.shuffle(self.files)
        else:
            for input, output in random.shuffle(self.dataset):
                yield input, output


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
    dl = DataLoader(dataset=datapipe, batch_size=128,
                    num_workers=1)

    model_name = get_model_name(config)
    print(model_name)

    load_embeddings(vocabulary=vocabulary, config=config)

    model = GPT(config)

    trainer = Trainer(model, vocabulary, config=config, lr=config.learning_rate)
    print('model', model)

    try:
        checkpoint = torch.load(f'studies/single-context/models/{model_name}-epoch-end.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optmizer_state_dict'])
        file = open(f'studies/single-context/models/{model_name}.log', "a")
        print('Model loaded')
    except FileNotFoundError:
        print('Model not found - training from scratch')
        pass
    
    if os.path.exists(f'studies/single-context/models/{model_name}.log'):
        file = open(f'studies/single-context/models/{model_name}.log', "a")
    else:
        file = open(f'studies/single-context/models/{model_name}.log', "w")        
        file.write(f'Initial weight:  {cfg.initial_weight}\n')
        file.write(f'Weight increase: {cfg.weight_increase}\n')
        file.write(f'Temp decrease:   {cfg.temp_decrease}\n')
        file.write(f'Neuron opening:  {cfg.neuron_opening}\n')
        file.write(f'History:  {history}\n')
        file.write(f'Next:     {next}\n')
        file.write(f'\n\nDataset:\n')
        for file_name in cfg.text_files:
            file.write(f'> {file_name}\n')
        
        file.write('----------------------------------------\n\n\n')

    try:
        os.remove(f'studies/single-context/input_report/report.log')
    except:
        pass
    
    input_report_file = open(f'studies/single-context/input_report/report.log', "x")
    input_report_file.write('INPUT REPORT')
    input_report_file.close()

    for epoch_idx in range(2):
        loss_all = 0
        batch_loss = 0
        model.train(True)
        acc_batchs = []
        input_report_file = open(f'studies/single-context/input_report/report.log', "a")
        input_report_file.write('\n\nNEW EPOCH\n')
        input_report_file.close()
        for (batch_idx, batch) in tqdm(enumerate(dl)):
            input_report_file = open(f'studies/single-context/input_report/report.log', "a")
            loss, acc = trainer.batch_train(batch)
            input_report_file.close()
            loss_all = loss_all + loss
            batch_loss = batch_loss + loss
    
            if batch_idx % 10 == 0:
                acc_batchs.append((batch_idx, acc))

            if batch_idx % 100 == 0 and batch_idx > 2:
                ckpt_path = os.path.join(
                    'studies/single-context/models/', f'{model_name}.pt')
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
            'studies/single-context/models/', f'{model_name}-epoch-end.pt')
        torch.save({'model_state_dict': model.state_dict(),
                    'optmizer_state_dict': trainer.optimizer.state_dict()
                    }, ckpt_path)
        
        print('')
        print('---------------------------------------------------------')
        print(f'%% Epoch {epoch_idx} loss: {batch_loss}\n')
        file = open(f'studies/single-context/models/{model_name}.log', "a")
        file.write(f'%% Epoch {epoch_idx} loss: {batch_loss}\n')
        file.close()
        print(f'Accuracy: {acc_batchs}')
        print('---------------------------------------------------------')


if __name__ == '__main__':
    train()
