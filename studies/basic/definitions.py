import re
import os
import json
import shutil
from src.context.recorder import Recorder
from src.context.context import Context
from src.context.vocabulary import Vocabulary


vocabulary = Vocabulary(include_start_end=True,
                        include_punctuation=False, accept_all=False)

record = False

if record:
    context = Recorder('test', vocabulary, initial_weight=0.1)
else:
    context = Context('test', vocabulary, initial_weight=0.1)

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


expand_from('car', 10)
expand_from('planet', 10)
expand_from('bottle', 10)

# context.prune_edges(1)

# vocabulary.accept_all = True
# context.add_text('boiled potato salad is a good recipe.')
if record:
    context.start_recording('output/test.gif', 'Test',
                            consider_stimulus=False, fps=2, arrow_size=0.02)

expand_from('car', 2)

context.stimulate('plastic')
context.stimulate('bottle')
# context.stimulate('engine')
print(context.get_top_stimuli(10))

record = False
if not record:
    context.render('output/test.png', 'Test',
                   consider_stimulus=False, figsize=(15, 15), arrow_size=0.05)
