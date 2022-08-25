import re
import os
import json
import shutil
from src.context.recorder import Recorder
from src.context.context import Context
from src.context.vocabulary import Vocabulary


vocabulary = Vocabulary(include_start_end=True,
                        include_punctuation=False, accept_all=False)

record = True

if record:
    context = Recorder('test', vocabulary,
                       initial_weight=0.2, temp_decrease=0.05)
else:
    context = Context('test', vocabulary,
                      initial_weight=0.2, temp_decrease=0.05)

definitions = json.loads(open('assets/kids_dictionary.json').read())


def expand_from(keyword, counter=3, added=None):
    if added is None:
        added = []

    if keyword in definitions.keys():
        added.append(keyword)
        defn = definitions[keyword]
        defs = defn['definitions']
        vars = defn['variations']

        if counter > 0:
            for definition in defs:
                missing, _ = context.add_definition(
                    keyword, definition)
                for missing_token in missing:
                    expand_from(missing_token, counter-1, added)

    return added

# expand_from('car', 1)
# expand_from('planet', 1)
expand_from('car', 1)
expand_from('wheel', 1)
expand_from('engine', 1)

# context.prune_edges(0.22)

if record:
    context.start_recording('output/test.gif', 'Test',
                            consider_stimulus=False,
                            fps=4,
                            arrow_size=0.02,
                            skip_empty_nodes=True)
    context.render_label_size = 0.1


if record:
    for i in range(0, 10):
        context.decrease_stimulus()

context.consider_stimulus = False

context.stimulate('plastic')
context.stimulate('bottle')


if record:
    for i in range(0, 10):
        context.decrease_stimulus()

record = False
if not record:
    context.render('output/test.png', 'Test',
                   consider_stimulus=False, figsize=(15, 15), arrow_size=0.05)
