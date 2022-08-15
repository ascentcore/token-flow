import re
from src.context.dataset import Dataset
from src.context.context import Context
from src.context.vocabulary import Vocabulary
from tqdm import tqdm

from src.net.trainer import Trainer

from src.net.models.autoencoder import AE
from src.net.models.residual import ResidualModel

record = True


def get_context(name,
                vocab,
                initial_weight=0.1,
                weight_increase=0.1,
                temp_decrease=0.1,
                neuron_opening=0.75):

    context = Context(name, vocab,
                      initial_weight=initial_weight,
                      neuron_opening=neuron_opening,
                      weight_increase=weight_increase,
                      temp_decrease=temp_decrease)

    return context


def read(execute):
    pattern = re.compile("^(wallace|biden|trump): (.*)")

    file = open('studies/political-debate/2020-debate-transcript-small.txt', 'r')
    last_speaker = None

    for line in tqdm(file.readlines()):
        line = line.strip().lower()

        result = pattern.search(line)

        if result is not None:
            last_speaker, text = result.groups()
            line = text

        execute(last_speaker, line)


def prepare_dataset():

    vocabulary = Vocabulary(
        accept_all=True, include_start_end=True, include_punctuation=False, use_lemma=False, add_lemma_to_vocab=False)
    dataset = Dataset(vocabulary)

    dataset.add_context(get_context('biden', vocabulary))
    dataset.add_context(get_context('trump', vocabulary))

    def prepare_context(last_speaker, line):
        if last_speaker != 'wallace':
            dataset.get_context(last_speaker).add_text(line)

    def prepare_dataset(last_speaker, line):
        if last_speaker != 'wallace':
            dataset.get_dataset(last_speaker).add_text(line)

    read(prepare_context)
    read(prepare_dataset)

    dataset.store('studies/political-debate/dataset')


def train():

    vocabulary = Vocabulary.from_file(
        'studies/political-debate/dataset', 'vocabulary.json')
    model = AE(vocabulary.size())
    # model = ResidualModel(vocabulary.size())

    trainer = Trainer(model, vocabulary, 'government top scientists', generate_length=50,
                      prevent_convergence_history=2)

    biden_context = Context.from_file(
        'studies/political-debate/dataset', 'biden', vocabulary)

    trump_context = Context.from_file(
        'studies/political-debate/dataset', 'biden', vocabulary)

    trainer.train(
        trump_context, 'studies/political-debate/dataset/biden.dataset.json', 50)

    # for i in range(0, 10):
    #     for context in [biden_context, trump_context]:
    #         trainer.train(
    #             context, 'studies/political-debate/dataset/biden.dataset.json', 50)


# prepare_dataset()
train()
