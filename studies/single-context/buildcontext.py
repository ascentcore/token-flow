from src.context.vocabulary import Vocabulary
from src.context.context import Context
from src.context.dataset import Dataset

import config


vocabulary = Vocabulary(
    accept_all=True,
    include_start_end=True,
    include_punctuation=True,
    use_lemma=False,
    add_lemma_to_vocab=False)


def get_context(name,
                initial_weight=config.initial_weight,
                weight_increase=config.weight_increase,
                temp_decrease=config.temp_decrease,
                neuron_opening=config.neuron_opening):

    context = Context(name, vocabulary,
                      initial_weight=initial_weight,
                      neuron_opening=neuron_opening,
                      weight_increase=weight_increase,
                      temp_decrease=temp_decrease)

    return context


def build_dataset():
    dataset = Dataset(vocabulary)
    dataset.delete_context('default')

    context = get_context('default')
    dataset.add_context(context)
    
    for text_file in config.text_files:
        text_data = open(config.dir_path + '/' + text_file).read()
        file_name = text_file.split('.')

        # context = get_context('default')
        # dataset.add_context(context)
        for line in text_data.splitlines():
            if line != '':
                context.add_text(line)

    print('Vocabulary size:', len(vocabulary.vocabulary))

    dataset.store('studies/text/dataset/')

    # context.render('studies/text/dataset/graph', force_text_rendering = True, dpi = 600)


def get_context_for_file(vocabulary, file_name):
    context = Context('custom', vocabulary)
    text_data = open(config.dir_path + '/' + file_name).read()
    for line in text_data.splitlines():
        if line != '':
            context.add_text(line)

    return context

if __name__ == '__main__':
    build_dataset()
    