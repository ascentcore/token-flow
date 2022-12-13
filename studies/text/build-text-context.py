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

    text_data = open(config.text_file).read()

    context = get_context('city_mouse')
    dataset.add_context(context)

    for line in text_data.splitlines():
        if line != '':
            context.add_text(line)

    print('Vocabulary size:', len(vocabulary.vocabulary))

    dataset.store('studies/text/dataset')


if __name__ == '__main__':
    build_dataset()
    