import numpy as np
import tqdm
from torchdata.datapipes.iter import IterDataPipe
from torch.utils.data import DataLoader

from config import text_file, contexts, size, history, next


def get_input(context):
    input = [[int(context.vocabulary.index_of(token)), stimulus] for token, stimulus in context.get_top_stimuli(size, history, next)]

    return input


class Text(IterDataPipe):

    def __init__(self, contexts, size):
        self.contexts = contexts
        self.size = size
        self.text_data = open(text_file).read()

    def __iter__(self):
        for context in self.contexts.values():
            context.decrease_stimulus(1)
        
        context = self.contexts['city_mouse']

        for phrase in self.text_data.splitlines():
            if phrase != '':
                input = get_input(context)

                _, sentences = context.vocabulary.get_token_sequence(
                    phrase, append_to_vocab=False, skip_eol=True)
                for sentence in sentences:
                    for tokens in sentence:
                        for token in tokens:
                            context.stimulate(token)
                            output = np.zeros(
                                context.vocabulary.size(), np.float32)
                            output[int(context.vocabulary.index_of(token))] = 1
                            yield input, output
                            input = get_input(context)

                context.stimulate_sequence(phrase, skip_eol=True)


def row_processer(row):
    print('row', row)

def train():
    datapipe = Text(contexts, size)
    datapipe = datapipe.map(row_processer)
    dl = DataLoader(dataset=datapipe, batch_size=4,
                    num_workers=4)

if __name__ == '__main__':
    train()
