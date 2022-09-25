import json
from src.context.dataset import Dataset
from src.context.context import Context
from src.context.vocabulary import Vocabulary
from src.net.ds.linear_dataset import LinearDataset

from src.net.trainer import Trainer

from src.net.models.autoencoder import AE
from src.net.models.residual import ResidualModel

vocabulary = Vocabulary(
    accept_all=True,
    include_start_end=True,
    include_punctuation=False,
    use_lemma=False,
    add_lemma_to_vocab=False)

initial_weight = 0.5
weight_increase = 0.037
temp_decrease = 0.25
neuron_opening = 0.75

lr = 1e-5


def get_context(name,
                initial_weight=initial_weight,
                weight_increase=weight_increase,
                temp_decrease=temp_decrease,
                neuron_opening=neuron_opening):

    context = Context(name, vocabulary,
                      initial_weight=initial_weight,
                      neuron_opening=neuron_opening,
                      weight_increase=weight_increase,
                      temp_decrease=temp_decrease)

    return context


def prepare_dataset():

    dataset = Dataset(vocabulary)
    dataset.from_folder('studies/various-texts/texts',
                        get_context, decrease_on_end=0)
    dataset.store('studies/various-texts/dataset')


def train_residual():

    vocabulary = Vocabulary.from_file(
        'studies/various-texts/dataset', 'vocabulary.json')
    # model = ResidualModel(vocabulary.size())
    model = AE(vocabulary.size())

    trainer = Trainer(model, vocabulary, lr=lr)

    settings = json.loads(
        open(f'studies/various-texts/dataset/dataset.settings.json').read())

    contexts = []

    for context_name in settings['contexts']:
        if context_name != 'default':
            context = Context.from_file(
                'studies/various-texts/dataset', context_name, vocabulary)
            contexts.append(context)

    pre = ""

    for iter in range(0, 100):
        print(f'Running iteration {iter}')
        for context in contexts:
            print(f'\n############ {context.name} ############')
            ds = LinearDataset(
                f'studies/various-texts/dataset/{context.name}.dataset.json')
            trainer.train(ds, epochs=50, batch_size=16)

            for c in contexts:
                c.decrease_stimulus(1)

                input_data = Dataset.get_sample_data(c, pre)
                text = trainer.get_sentence(
                    c, input_data, generate_length=150, prevent_convergence_history=5, num_patches=None)
                print(f'\n\n{c.name}: [{pre}] {text}')
        # torch.save(model, f'studies/various-texts/models/model_{iter}')


if __name__ == '__main__':
    prepare_dataset()
    train_residual()
