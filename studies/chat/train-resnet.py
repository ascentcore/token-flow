import json
from math import nan
import torch
import time
import os
from tqdm import tqdm
from torchdata.datapipes.iter import IterDataPipe
import numpy as np
from torch.utils.data import DataLoader
from src.context.context import Context
from src.context.dataset import Dataset

from src.context.vocabulary import Vocabulary

from src.net.models.residual import ResidualModel
from src.net.models.autoencoder import AE
from src.net.trainer import Trainer
import config as config


class RuntimeDataPipe(IterDataPipe):
    def __init__(self, contexts):
        self.contexts = contexts
        chat_data = json.loads(open(config.file).read())
        self.chat_data = []
        for id in tqdm(list(chat_data.keys())[config.start_index:config.end_index]):
            content = chat_data[id]["content"]
            for data in content:
                agent = data["agent"]
                message = data["message"]
                self.chat_data.append((agent, message))
            self.chat_data.append((None, None))

        print(f'Chat data size: {len(self.chat_data)}')

    def __iter__(self):
        for (agent, message) in self.chat_data:
            if agent is None:
                for context in self.contexts.values():
                    context.decrease_stimulus(1)
                continue

            for context_key in self.contexts.keys():
                if context_key != agent:
                    self.contexts[context_key].stimulate_sequence(message)

            context = self.contexts[agent]
            _, sentences = context.vocabulary.get_token_sequence(
                message, append_to_vocab=False)
            input = context.get_stimuli()
            for sentence in sentences:
                for tokens in sentence:
                    for token in tokens:
                        context.stimulate(token)
                        output = context.get_stimuli()
                        if config.filtered_output:
                            filtered_output = [
                                1 if x == 1 else 0 for x in output]
                            yield input, filtered_output
                        else:
                            yield input, output
                        input = output
                if config.decrease_on_end != None:
                    input = context.get_stimuli()
                    context.decrease_stimulus(config.decrease_on_end)


def row_processer(row):
    return [np.array(row[0], np.float32), np.array(row[1], np.float32)]


def train():

    vocabulary = Vocabulary.from_file(
        'studies/chat/dataset', 'vocabulary.json')
    res_model = ResidualModel(vocabulary.size(), 10)
    
    try:
        if config.retrain == True:
            model = res_model
            print('Retraining model...')
        else:
            model = torch.load(f'studies/chat/models/model.pt')
            print('Model loaded succesfully')
    except:
        model = res_model
        print('Model created succesfully')

    try:
        os.mkdir('studies/chat/models/')
    except:
        pass

    trainer = Trainer(model, vocabulary, lr=0.001)

    settings = json.loads(
        open(f'studies/chat/dataset/dataset.settings.json').read())

    contexts_list = None
    contexts = {}

    for context_name in settings['contexts']:
        if context_name != 'default' and (contexts_list is None or context_name in contexts_list):
            context = Context.from_file(
                'studies/chat/dataset', context_name, vocabulary)
            contexts[context_name] = context

    def test_generation():
        print('Generating text')
        f = open("./chat.txt", "w")
        model.train(False)
        # stimulate_with = "Hello, do you like cats?"
        stimulate_with = None
        for c in contexts.values():
            c.decrease_stimulus(1)
        for _ in range(30):
            for c in contexts.values():
                if stimulate_with != None:
                    c.stimulate_sequence(stimulate_with)
                text = trainer.get_sentence(
                    c, [], generate_length=20, prevent_convergence_history=1,
                    break_on_eol=True)
                stimulate_with = text
                f.write(f'{c.name}: {text}\n')

        f.close()

        for c in contexts.values():
            c.decrease_stimulus(1)

    # test_generation()

    datapipe = RuntimeDataPipe(contexts)
    datapipe = datapipe.map(row_processer)
    dl = DataLoader(dataset=datapipe, batch_size=32, num_workers=1)

    for i in range(1000):
        loss_all = 0
        model.train(True)
        for (batch_idx, batch) in tqdm(enumerate(dl)):
            loss = trainer.batch_train(batch)
            loss_all = loss_all + loss

        if loss_all == nan:
            print('Loss is nan, stopping training')
            break

        test_generation()
        print(f'Epoch {i} loss: {loss_all}')
        torch.save(model, f'studies/chat/models/model.pt')


if __name__ == '__main__':
    train()
