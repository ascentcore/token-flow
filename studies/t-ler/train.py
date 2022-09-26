import os
import torch
import shutil
from tqdm import tqdm

import torchdata.datapipes as dp
import numpy as np
from torch.utils.data import DataLoader

from src.context.dataset import BasicInMemDataset, Dataset

from src.context.vocabulary import Vocabulary
from src.context.context import Context
from src.net.ds.linear_runtime_inmem_dataset import LinearRuntimeInMemDataset

from src.net.models.autoencoder import AE
from src.net.trainer import Trainer

import config as cfg


def get_context(name, vocabulary):

    context = Context(name, vocabulary,
                      initial_weight=cfg.initial_weight,
                      neuron_opening=cfg.neuron_opening,
                      weight_increase=cfg.weight_increase,
                      temp_decrease=cfg.temp_decrease)

    return context


def create_dataset():
    vocabulary = Vocabulary(
        accept_all=True,
        include_start_end=True,
        include_punctuation=False,
        use_lemma=False,
        add_lemma_to_vocab=False)

    path = f'studies/t-ler/data/{cfg.folder}'
    data_path = f'{path}/dataset'

    try:
        shutil.rmtree(data_path)
    except:
        pass

    os.mkdir(data_path)
    os.mkdir(f'{data_path}/datasets')
    os.mkdir(f'{data_path}/contexts')
    os.mkdir(f'{data_path}/models')

    if cfg.render_context:
        os.mkdir(f'{data_path}/renders')

    # Build vocabulary from file (contexts will be built on the fly)
    vocabulary.from_folder(f'{path}/train')
    vocabulary.from_folder(f'{path}/test')
    vocabulary.save_vocabulary(f'{data_path}')

    for (dir_path, dir_names, file_names) in os.walk(f'{path}/train'):
        for file_name in file_names:
            if file_name.endswith('.txt'):
                print(f'Bulding context for {file_name}')
                train_context = get_context(file_name, vocabulary)
                train_context.load_text_file(os.path.join(dir_path, file_name))
                train_context.store(f'{data_path}/contexts')

                if cfg.render_context:
                    train_context.render(
                        f'{data_path}/renders/{train_context.name}.png', train_context.name, consider_stimulus=False, skip_empty_nodes=True)

                print(f'Building dataset for {file_name}')
                ds = BasicInMemDataset(train_context)

                with open(os.path.join(dir_path, file_name), 'r') as f:
                    text = f.read()
                    ds.add_text(text, decrease_on_end=cfg.decrease_on_end,
                                filtered_output=cfg.filtered_output)

                    ds.to_csv(
                        f'{data_path}/datasets/{train_context.name}.dataset.csv')

    return vocabulary


def restore_vocabulary():
    vocabulary = Vocabulary.from_file(
        f'studies/t-ler/data/{cfg.folder}', 'vocabulary.json')
    return vocabulary


def row_processer(row):
    full_row = np.array(row, np.float32)
    return np.array_split(full_row, 2)


if __name__ == '__main__':
    if cfg.create_dataset == True:
        vocabulary = create_dataset()
    else:
        vocabulary = restore_vocabulary()

    model = AE(vocabulary.size())
    trainer = Trainer(model, vocabulary, lr=cfg.lr)

    test_context = get_context('test', vocabulary)
    test_context.load_text_file(
        f'studies/t-ler/data/{cfg.folder}/test/test.txt')

    datapipe = dp.iter.FileLister(
        f'studies/t-ler/data/{cfg.folder}/dataset/datasets')
    datapipe = datapipe.open_files(mode='rb')
    datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1)
    datapipe = datapipe.map(row_processer)
    dl = DataLoader(dataset=datapipe, batch_size=cfg.batch_size, num_workers=2)

    for i in range(cfg.epochs):
        loss_all = 0
        for (batch_idx, batch) in tqdm(enumerate(dl)):
            loss = trainer.batch_train(batch)
            loss_all = loss_all + loss

        print(loss_all)

        text = trainer.get_sentence(
            test_context, [], generate_length=cfg.generate_size,
            prevent_convergence_history=cfg.prevent_convergence_history)

        print('#######################################')
        print(text)
        print('#######################################')
        torch.save(
            model, f'studies/t-ler/data/{cfg.folder}/dataset/models/model_{i}.pt')
