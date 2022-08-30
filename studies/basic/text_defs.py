import re
import os
import json
import shutil
from src.context.recorder import Recorder
from src.context.context import Context
from src.context.vocabulary import Vocabulary


vocabulary = Vocabulary(include_start_end=True,
                        include_punctuation=False,
                        accept_all=False)

record = False

if record:
    context = Recorder('test', vocabulary,
                       initial_weight=0.5, definition_weight=0.1, temp_decrease=0.05)
else:
    context = Context('test', vocabulary,
                      initial_weight=0.5, definition_weight=0.1, temp_decrease=0.05)

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


if record:
    context.start_recording('output/test.gif', 'Test',
                            consider_stimulus=False,
                            fps=4,
                            arrow_size=0.02,
                            skip_empty_nodes=True)
    context.render_label_size = 0.1


context.add_text('my favorite coffee is dark roast.')
context.add_text('my favorite drink is beer.')
# context.add_text('my favorite outfit is a black shirt.')
# expand_from('coffee', 1)
# context.stimulate_sequence('what is your favorite drink?')
context.stimulate('favorite')
context.stimulate('drink')

# print(context.get_top_stimuli(10))

print(context.get_stimuli())

record = False
if not record:
    context.render('output/test.png', 'Test',
                   consider_stimulus=False, figsize=(5, 5), arrow_size=2)
