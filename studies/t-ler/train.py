import os
from src.context.dataset import BasicInMemDataset, Dataset

from src.context.vocabulary import Vocabulary
from src.context.context import Context
from src.net.ds.linear_runtime_inmem_dataset import LinearRuntimeInMemDataset

from src.net.models.autoencoder import AE
from src.net.trainer import Trainer

import config as cfg


def build_vocabulary():
    vocabulary = Vocabulary(
        accept_all=True,
        include_start_end=True,
        include_punctuation=False,
        use_lemma=False,
        add_lemma_to_vocab=False)

    # Build vocabulary from file (contexts will be built on the fly)
    vocabulary.from_folder(f'studies/t-ler/data/{cfg.folder}/train')
    vocabulary.from_folder(f'studies/t-ler/data/{cfg.folder}/test')
    vocabulary.save_vocabulary(f'studies/t-ler/data/{cfg.folder}')

    return vocabulary


def restore_vocabulary():
    vocabulary = Vocabulary.from_file(
        f'studies/t-ler/{cfg.folder}', 'vocabulary.json')
    return vocabulary


def get_context(name, vocabulary):

    context = Context(name, vocabulary,
                      initial_weight=cfg.initial_weight,
                      neuron_opening=cfg.neuron_opening,
                      weight_increase=cfg.weight_increase,
                      temp_decrease=cfg.temp_decrease)

    return context


if __name__ == '__main__':
    vocabulary = build_vocabulary()
    # vocabulary = restore_vocabulary()

    model = AE(vocabulary.size())
    trainer = Trainer(model, vocabulary, lr=cfg.lr)

    test_context = get_context('test', vocabulary)
    test_context.load_text_file(
        f'studies/t-ler/data/{cfg.folder}/test/test.txt')

    test_ds = BasicInMemDataset(test_context)
    test_ds.add_text(cfg.pre)

    # test_ds.pretty_print()

    for i in range(cfg.repeats):
        print(f'Running repeatable {i}')
        for (dir_path, dir_names, file_names) in os.walk(f'studies/t-ler/data/{cfg.folder}/train'):
            for file_name in file_names:
                if file_name.endswith('.txt'):
                    print(f'Bulding context for {file_name}')
                    train_context = get_context(file_name, vocabulary)
                    train_context.load_text_file(
                        os.path.join(dir_path, file_name))

                    if cfg.render_context and i == 0:
                        train_context.render(
                            f'studies/t-ler/data/{cfg.folder}/{train_context.name}.png', train_context.name, consider_stimulus=False, skip_empty_nodes = True)

                    print(f'Building dataset for {file_name}')
                    ds = BasicInMemDataset(train_context)

                    with open(os.path.join(dir_path, file_name), 'r') as f:
                        text = f.read()
                        ds.add_text(text, decrease_on_end=cfg.decrease_on_end,
                                    filtered_output=cfg.filtered_output)

                        # ds.pretty_print()

                    dataset = LinearRuntimeInMemDataset(ds)
                    trainer.train(dataset, epochs=cfg.epochs,
                                  batch_size=cfg.batch_size)

                    text = trainer.get_sentence(
                        test_context, test_ds.data, generate_length=cfg.generate_size,
                        prevent_convergence_history=cfg.prevent_convergence_history)

                    print('#######################################')
                    print(text)
                    print('#######################################')
