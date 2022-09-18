import os
import re

from tqdm import tqdm
from src.context.dataset import Dataset
from src.context.context import Context
from src.context.vocabulary import Vocabulary


vocabulary = Vocabulary(
    accept_all=True,
    include_start_end=True,
    include_punctuation=False,
    use_lemma=False,
    add_lemma_to_vocab=False)

initial_weight = 0.5
weight_increase = 0.037
temp_decrease = 0.08
neuron_opening = 0.75


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


def build_dataset():

    dataset = Dataset(vocabulary)

    regex = r"^([a-zA-Z ]+):(.*)$"  # character: line

    res = []
    path = 'studies/south-park/scripts'
    for (dir_path, dir_names, file_names) in os.walk(path):
        res.extend(file_names)

    textual_dataset = []
    current_text_content = {
        'contexts': [],
        'lines': []
    }

    for name in res:
        source = open(f'{path}/{name}', 'r')
        lines = source.readlines()

        for line in lines:
            line = line.strip() # remove whitespace
            if line != '':
                if re.match(r'^\[.*\]$', line):
                    if len(current_text_content['lines']) > 0:
                        textual_dataset.append(current_text_content)
                        current_text_content = {
                            'contexts': [],
                            'lines': []
                        }
                else:
                    matches = re.findall(regex, line)
                    if len(matches) > 0:
                        character, char_line = matches[0]
                        char_line = char_line.strip()

                        if character not in current_text_content['contexts']:
                            current_text_content['contexts'].append(character)

                        current_text_content['lines'].append((character, char_line))

        if len(current_text_content['lines']) > 0:
            textual_dataset.append(current_text_content)            

    
    ## Prepare contexts for all characters
    print('Computing context graph ...')
    for text_content in tqdm(textual_dataset):
        contexts = text_content['contexts']
        lines = text_content['lines']

        for context in contexts:
            if not dataset.has_context(context):
                context = get_context(context)
                dataset.add_context(context)
            else:
                context = dataset.get_context(context)
            
        for line in lines:
            character, char_line = line
            context = dataset.get_context(character)
            context.add_text(char_line)

    ## Prepare dataset 
    print('Preparing dataset...')
    for text_content in tqdm(textual_dataset):
        contexts = text_content['contexts']
        lines = text_content['lines']

        current_contexts = [dataset.get_context(name) for name in contexts]
        for line in lines:
            character, char_line = line
            
            for context in current_contexts:
                # if context.name != character:
                #     dataset.get_dataset(context.name, context.name).add_text(f'{character} {char_line}')
                # else:
                dataset.get_dataset(context.name, context.name).add_text(char_line)

        dataset.reset_stimulus()

    dataset.store('studies/south-park/dataset')      
            


if __name__ == '__main__':
    build_dataset()
