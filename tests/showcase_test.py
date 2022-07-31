from src.recorder import Recorder
from src.context import Context
from src.vocabulary import Vocabulary

import unittest
import json


record = True


class TestShowcase(unittest.TestCase):

    def test_kids_dictionary(self):

        initial_weight = 0.2

        if record:
            context = Recorder('Showcase', include_start=False,
                               initial_weight=initial_weight)
        else:
            context = Context('Showcase', include_start=False,
                              initial_weight=initial_weight)

        definitions = json.loads(open('assets/kids_dictionary.json').read())
        keys = definitions.keys()

        def add_definition_for(keyword):
            if keyword in keys:
                for definition in definitions[keyword]['definitions']:
                    context.add_definition(keyword, definition)

        if record:
            context.start_recording('output/tests/output.gif',
                                    title="Test Gif", consider_stimulus=False, fps=3, arrow_size=0.5)

        # for key in definitions.keys():
        #     add_definition_for(key)

        for key in ['tomato', 'olive', 'oil', 'vegetable', 'salad', 'cucumber', 'lettuce', 'eat', 'food', 'tomato']:
            add_definition_for(key)

        if record:
            context.stop_recording()
        else:
            context.render('output/tests/output.png',
                           consider_stimulus=False, figsize=(20, 20))

    def test_showcase(self):

        if record:
            context = Recorder(
                'Showcase', include_start=False, initial_weight=0.2)
        else:
            context = Context('Showcase', include_start=False,
                              initial_weight=0.2)

        if record:
            context.start_recording('output/tests/output.gif',
                                    title="Test Gif", consider_stimulus=False, fps=3, arrow_size=0.5)

        context.add_definition(
            'salad', 'a mixture of cold vegetables such as lettuce, tomato, and cucumber, served with a dressing.')

        context.add_definition(
            'lettuce', 'a plant with large, crisp leaves that can be eaten.'
        )

        context.add_definition(
            'tomato', 'a red or yellow fruit with a juicy pulp. A tomato is eaten either raw or cooked as a vegetable.')
        context.add_definition(
            'cucumber', 'a plant with a green, round, fleshy fruit that is used in salads, soups, and other dishes.')
        context.add_definition(
            'vegetbale', 'a plant or part of a plant, such as carrots, beans, or lettuce, that is used for food.')

        # context.add_definition('sausage', 'a mixture of chopped meat and spices stuffed into a casing of animal intestine.')

        context.add_definition('favorite', 'food is salad')
        context.add_definition(
            'greek', 'salad olive oil, tomatoes, lettuce, and cucumber')
        context.add_definition(
            'tomato', 'a red or yellow fruit with a juicy pulp')
        context.add_definition(
            'sausage', 'a mixture of chopped meat and spices stuffed')
        context.add_definition('eat', 'the process of eating food')

        context.add_text('my favorite food are meats')
        context.add_text('my favorite food are fruits')
        context.add_text('I like greek salad')
        context.add_text('my prefference in food are vegetables')
        context.add_text('I do not eat meat')

        context.consider_stimulus = True

        context.stimulate_sequence(
            'I am very hungry, I need to eat, I think I need a salad or something lite')

        context.render('output/tests/output.png', consider_stimulus=True)

        if record:
            context.stop_recording()


if __name__ == '__main__':
    unittest.main()
