import os
import torch
import numpy as np

from src.context.vocabulary import Vocabulary
from src.context.context import Context

from src.net.trainer import Trainer

import config as cfg


def get_context(name, vocabulary):

    context = Context(name, vocabulary,
                      initial_weight=cfg.initial_weight,
                      neuron_opening=cfg.neuron_opening,
                      weight_increase=cfg.weight_increase,
                      temp_decrease=cfg.temp_decrease)

    return context


def restore_vocabulary():
    vocabulary = Vocabulary.from_file(
        f'studies/t-ler/data/{cfg.folder}/dataset', 'vocabulary.json')
    return vocabulary


def row_processer(row):
    full_row = np.array(row, np.float32)
    input = full_row[:-1]
    out_idx = full_row[-1]

    output = np.zeros(len(input), np.float32)
    output[int(out_idx)] = 1

    return [input, output]


if __name__ == '__main__':

    path = f'studies/t-ler/data/{cfg.folder}'

    vocabulary = restore_vocabulary()

    model = torch.load(f'studies/t-ler/data/{cfg.folder}/dataset/models/model.pt')
    model.eval()        

    
    trainer = Trainer(model, vocabulary, lr=cfg.lr)

    test_contexts = []

    for (dir_path, dir_names, file_names) in os.walk(f'{path}/test'):
        for file_name in file_names:
            if file_name.endswith('.txt'):
                test_context = get_context(file_name, vocabulary)
                test_context.load_text_file(os.path.join(dir_path, file_name))
                test_contexts.append(test_context)

  
    for context in test_contexts:
        context.decrease_stimulus(1)
        context.stimulate_sequence(cfg.pre)
        text = trainer.get_sentence(
            context, [], generate_length=cfg.generate_size,
            prevent_convergence_history=cfg.prevent_convergence_history)

        print('#######################################')
        print(text)
        print('#######################################')

