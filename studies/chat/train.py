import re
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


def get_context(name,
                vocab,
                initial_weight=0.05,
                weight_increase=0.01,
                temp_decrease=0.01,
                neuron_opening=0.95):

    context = Context(name, vocab,
                      initial_weight=initial_weight,
                      neuron_opening=neuron_opening,
                      weight_increase=weight_increase,
                      temp_decrease=temp_decrease)

    return context


def read(fn, authors=None):
    regex = r"([a-zA-Z0-9^:]+)*:(.*)"

    contexts = []
    lines = []
    print(f'Preparing context for {fn}')
    file = open(f'studies/chat/texts/{fn}', 'r')
    for line in tqdm(file.readlines()):
        line = line.strip().lower()
        if re.match(regex, line):

            matches = re.finditer(regex, line)

            for matchNum, match in enumerate(matches, start=1):
                author = match.group(1)
                text = match.group(2)

                if author not in contexts and (authors is None or author in authors):
                    contexts.append(author)

            if authors is None or author in authors:
                lines.append((author, text))
        else:
            print('>>>', line)

    return contexts, lines


def prepare_dataset():

    vocabulary = Vocabulary(
        accept_all=True,
        include_start_end=True,
        include_punctuation=True,
        use_lemma=False,
        add_lemma_to_vocab=False)

    dataset = Dataset(vocabulary)

    res = []
    for (dir_path, dir_names, file_names) in os.walk('studies/chat/texts/'):
        res.extend(file_names)

    for file_name in res:
        contexts, lines = read(file_name)

        for context_name in contexts:
            if not dataset.has_context(context_name):
                print('Creating context: ', context_name)
                dataset.add_context(get_context(context_name, vocabulary))
            # else:
            #     dataset.get_context(context_name).decrease_stimulus(1)

        # create own graph
        for context_name, text in tqdm(lines):
            dataset.get_context(context_name).add_text(text)

        for context in contexts:
            dataset.get_context(context).render(
                f'output/{context}.png', context, False, arrow_size=3, skip_empty_nodes=True)

        # generate dataset
        for context_name, text in tqdm(lines):
            dataset.get_dataset(context_name, context_name).add_text(text)

            for rest in [context for context in contexts if context != context_name]:
                dataset.get_context(rest).stimulate_sequence(text)

    dataset.store('studies/chat/dataset')


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
                context, f'studies/chat/dataset/{context.name}.dataset.json', 150)

            for c in contexts:
                c.decrease_stimulus(1)

            for i in range(0, 10):
                responder = contexts[0] if i % 2 == 0 else contexts[1]
                listener = contexts[1] if i % 2 == 0 else contexts[0]

                text = trainer.get_sentence(
                    responder, generate_length=10)
                print(f'{responder.name}: {text}')
                listener.stimulate_sequence(text)


# prepare_dataset()
train()
