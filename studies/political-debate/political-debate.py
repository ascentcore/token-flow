import re
import os
import sys
import shutil
from src.recorder import Recorder
from src.context import Context
from src.vocabulary import Vocabulary
from tqdm import tqdm

record = True


def get_context(name, record=record,
                initial_weight=0.2,
                weight_increase=0.05,
                temp_decrease=0.025,
                arrow_size=0.01,
                neuron_opening=0.8,
                one_way=False):

    if record:
        context = Recorder(name, include_start=False,
                           initial_weight=initial_weight, neuron_opening=neuron_opening, weight_increase=weight_increase, temp_decrease=temp_decrease)
    else:
        context = Context(name, include_start=False,
                          initial_weight=initial_weight, neuron_opening=neuron_opening, weight_increase=weight_increase, temp_decrease=temp_decrease)

    return context


def read(execute):
    pattern = re.compile("^(wallace|biden|trump): (.*)")

    file = open('studies/political-debate/2020-debate-transcript.txt', 'r')
    last_speaker = None

    for line in tqdm(file.readlines()):
        line = line.strip().lower()

        result = pattern.search(line)

        if result is not None:
            last_speaker, text = result.groups()
            line = text

        execute(last_speaker, line)


def prepare_contexts():
    contexts = {
        "wallace": get_context('wallace'),
        "biden": get_context('biden'),
        "trump": get_context('trump')
    }

    def exec(last_speaker, line):
        contexts[last_speaker].add_text(line, accept_all=False)

    read(exec)

    try:
        shutil.rmtree('studies/political-debate/contexts')                
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    os.mkdir('studies/political-debate/contexts')

    contexts["trump"].store('studies/political-debate/contexts')
    contexts["biden"].store('studies/political-debate/contexts')

    return contexts


def process(contexts):

    trump = contexts['trump']
    biden = contexts['biden']

    # trump.prune_edges(0.12)
    # biden.prune_edges(0.12)

    trump.start_recording('output/trump.gif', 'Trump', consider_stimulus=True,
                          skip_empty_nodes=True, fps=3, arrow_size=.02, force_text_rendering=False)
    biden.start_recording('output/biden.gif', 'Biden', consider_stimulus=True,
                          skip_empty_nodes=True, fps=3, arrow_size=.02, force_text_rendering=False)

    sentence = 'The economy is, I think it’s fair to say, recovering faster than expected from the shutdown'

    trump.stimulate_sequence(sentence)
    biden.stimulate_sequence(sentence)

    for _ in range(40):
        trump.decrease_stimulus()
        biden.decrease_stimulus()

    trump.stop_recording()
    biden.stop_recording()
    # def exec(last_speaker, line):
    #     if last_speaker == 'wallace':
    #         contexts["trump"].stimulate_sequence(line)

    # read(exec)

def load():
    contexts = {
        "biden": get_context('biden'),
        "trump": get_context('trump')
    }

    contexts["trump"].load('studies/political-debate/contexts')
    contexts["biden"].load('studies/political-debate/contexts')

    trump = contexts['trump']
    biden = contexts['biden']

    trump.prune_edges(0.12)
    biden.prune_edges(0.12)

    trump.start_recording('output/trump.gif', 'Trump', consider_stimulus=True,
                          skip_empty_nodes=True, fps=3, arrow_size=.02, force_text_rendering=False)
    biden.start_recording('output/biden.gif', 'Biden', consider_stimulus=True,
                          skip_empty_nodes=True, fps=3, arrow_size=.02, force_text_rendering=False)

    sentence = 'The economy is, I think it’s fair to say, recovering faster than expected from the shutdown'

    trump.stimulate_sequence(sentence)
    biden.stimulate_sequence(sentence)

    for _ in range(40):
        trump.decrease_stimulus()
        biden.decrease_stimulus()

    trump.stop_recording()
    biden.stop_recording()


# prepare_contexts()
load()
# process(prepare_contexts())
