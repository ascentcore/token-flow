import torch
import os
import json
from src.context.dataset import Dataset
from src.context.context import Context
from src.context.recorder import Recorder
from src.context.vocabulary import Vocabulary
from tqdm import tqdm

from src.net.trainer import Trainer

from src.net.models.autoencoder import AE
from src.net.models.residual import ResidualModel

record = False


def get_context(name,
                vocab,
                initial_weight=0.5,
                weight_increase=0.1,
                temp_decrease=0.05,
                neuron_opening=0.95):

    cls = Recorder if record else Context

    context = cls(name, vocab,
                  initial_weight=initial_weight,
                  neuron_opening=neuron_opening,
                  weight_increase=weight_increase,
                  temp_decrease=temp_decrease)

    if record:
        context.start_recording(
            'studies/chat/recording.gif', 'Record', True)

    return context


def read(execute, fn):
    print(f'Preparing context for {fn}')
    file = open(f'studies/chat/texts/{fn}', 'r')
    for line in tqdm(file.readlines()):
        line = line.strip().lower()
        execute(fn, line)


def prepare_dataset():

    vocabulary = Vocabulary(
        accept_all=True,
        include_start_end=True,
        include_punctuation=False,
        use_lemma=False,
        add_lemma_to_vocab=False)
    dataset = Dataset(vocabulary)

    if record:
        dataset.contexts['default'] = get_context('default', vocabulary)

    def prepare_context(context_name, line):
        if not dataset.has_context(context_name):
            print('Creating context: ', context_name)
            dataset.add_context(get_context(context_name, vocabulary))

        dataset.get_context(context_name).add_text(line)

    def prepare_dataset(context_name, line):
        dataset.get_dataset(context_name, context_name).add_text(line)

    res = []
    for (dir_path, dir_names, file_names) in os.walk('studies/chat/texts/'):
        res.extend(file_names)

    res = ['trump.txt']

    for fn in res:
        read(prepare_context, fn)

    for fn in res:
        read(prepare_dataset, fn)

    dataset.store('studies/chat/dataset')

    if record:
        dataset.contexts['default'].stop_recording()


def train():

    vocabulary = Vocabulary.from_file(
        'studies/chat/dataset', 'vocabulary.json')
    model = AE(vocabulary.size())
    # model = ResidualModel(vocabulary.size())

    trainer = Trainer(model, vocabulary)

    settings = json.loads(
        open(f'studies/chat/dataset/dataset.settings.json').read())

    contexts = []

    for context_name in settings['contexts']:
        if context_name != 'default':
            context = Context.from_file(
                'studies/chat/dataset', context_name, vocabulary)
            contexts.append(context)

            

    for iter in range(0, 100):
        for context in contexts:
            print(f'############ {context.name} ############')
            trainer.train(
                context, f'studies/chat/dataset/{context.name}.dataset.json', 50)
            print('------------------------------------------')
            trainer.generate(context, '', generate_length=40,
                             prevent_convergence_history=10)
            print('------------------------------------------')

        # torch.save(model, f'studies/chat/models/model_{iter}')


prepare_dataset()
train()
