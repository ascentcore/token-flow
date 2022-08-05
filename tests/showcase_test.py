from concurrent.futures import process
from src.recorder import Recorder
from src.context import Context
from src.vocabulary import Vocabulary
from tqdm import tqdm
import unittest
import json


def get_context(name, record=False, initial_weight=0.01,
                weight_increase=0.01,
                temp_decrease=0.01,
                arrow_size=0.01,
                neuron_opening=0.9,
                one_way=False):

    if record:
        context = Recorder(name, include_start=False,
                           initial_weight=initial_weight, neuron_opening=neuron_opening, weight_increase=weight_increase, temp_decrease=temp_decrease)
    else:
        context = Context(name, include_start=False,
                          initial_weight=initial_weight, neuron_opening=neuron_opening, weight_increase=weight_increase, temp_decrease=temp_decrease)

    return context

class TestShowcase(unittest.TestCase):

    def test_simple(self):

        record = False

        context = get_context(
            'Showcase', initial_weight=0.1, weight_increase=0.1, temp_decrease=0.1, record=record)

        ## Ground truth started ##
        if record:
            context.start_recording('output/tests/output.gif',
                                    title="", skip_empty_nodes=True, force_text_rendering=True, consider_stimulus=True, fps=2, arrow_size=2)

        # context.add_definition('tomato', 'a fruit, usually red')
        # context.add_definition('steak', 'a meaty cut of meat')

        # context.add_definition('favorite', 'a person who is fond of something')
        # context.add_definition('food', 'a thing that is eaten')

        ## Ground truth ended ##

        # context.add_text('my favorite food is tomato')
        # context.add_text('my favorite food is steak')
        context.add_text(biden)

        context.prune_edges(0.12)

        context.stimulate('war')
        context.stimulate('today')

        print(context.get_stimuli())
        print(context.get_matrix())

        # for i in range(20):
        #     context.decrease_stimulus()

        if record:
            context.stop_recording()

        context.render('output/tests/output.png',
                       consider_stimulus=True, 
                       figsize=(14, 14), 
                       skip_empty_nodes=True, 
                       arrow_size=.3, 
                       force_text_rendering=False)

    def test_basic_start(self):
        record = True
        definitions = json.loads(open('assets/kids_dictionary.json').read())

        context = get_context(
            'Showcase', initial_weight=0.2, weight_increase=0.1, record=record, temp_decrease=0.1)

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

        # text = 'a person who does not eat or does not believe in eating meat, fish, fowl, or, in some cases, any food derived from animals, as eggs or cheese, but subsists on vegetables, fruits, nuts, grain'
        # vegetable = 'any plant whose fruit, seeds, roots, tubers, bulbs, stems, leaves, or flower parts are used as food, as the tomato, bean, beet, potato, onion, asparagus, spinach, or cauliflower'
        # salad = 'a usually cold dish consisting of vegetables, as lettuce, tomatoes, and cucumbers, covered with a dressing and sometimes containing seafood, meat, or eggs'
        # greek_salad = 'Greek salad or horiatiki salad  is a popular salad in Greek cuisine generally made with pieces of tomatoes, cucumbers, onion, feta cheese (usually served as a slice on top of the other ingredients), and olives (typically Kalamata olives) and dressed with salt, Greek oregano, and olive oil. Common additions include green bell pepper slices or caper berries (especially on the Dodecanese islands). Greek salad is often imagined as a farmer breakfast or lunch, as its ingredients resemble those that a Greek farmer might have on hand.'

        # def process_text(definition, text, depth=2):
        #     missing, _ = context.add_definition(definition, text)
        #     for token in missing:
        #         added = expand_from(token, depth)
        #         print(added)

        # process_text('vegetarian', text, 4)
        # process_text('vegetable', vegetable, 4)
        # process_text('salad', salad, 4)
        # process_text('greek', greek_salad, 4)

        expand_from('car', 4)
        # expand_from('car', 2)

        # context.prune_edges(0.32)

        # context.stimulate('race')
        # context.stimulate('service')

        if record:
            context.start_recording('output/tests/output.gif',
                                    title="Test Gif", skip_empty_nodes=True, consider_stimulus=True, fps=2, arrow_size=2)

        # context.stimulate_sequence(greek_salad)

        context.stimulate('car')
        context.stimulate('engine')
        context.stimulate('move')
        context.stimulate('wheel')
        context.stimulate('land')
        context.stimulate('place')

        context.render('output/tests/output.png',
                       consider_stimulus=True, figsize=(8, 8), skip_empty_nodes=True, arrow_size=3)

        if record:
            context.stop_recording()

    def test_kids_dictionary(self):

        initial_weight = 0.01
        weight_increase = 0.01
        arrow_size = 0.01
        neuron_opening = 0.9
        consider_stimulus = True
        one_way = False
        record = True

        if record:
            context = Recorder('Showcase', include_start=False,
                               initial_weight=initial_weight, neuron_opening=neuron_opening, weight_increase=weight_increase)
        else:
            context = Context('Showcase', include_start=False,
                              initial_weight=initial_weight, neuron_opening=neuron_opening, weight_increase=weight_increase)

        definitions = json.loads(open('assets/kids_dictionary.json').read())
        keys = definitions.keys()

        def add_definition_for(keyword):
            if keyword in keys:
                for definition in definitions[keyword]['definitions']:
                    context.prepare_vocabulary(definition)
                    # context.add_definition(
                    #     keyword, definition, one_way=one_way)

        text = """
            A traditional Greek salad consists of sliced cucumbers, tomatoes, green bell pepper, red onion, olives, and feta cheese. This classic combination is delicious, so I stick to it, just adding a handful of mint leaves for a fresh finishing touch.
            My olives of choice are Kalamata olives. Commonly used in Greek food, their salty, briny flavor is delectable alongside the feta and crisp veggies. Instead of slicing large tomatoes, I use cherry tomatoes because they release less water into the salad than larger tomatoes would. I also seed my cucumber to avoid making my salad watery.
        """

        # _, sequences = context.vocabulary.get_token_sequence(text)

        # for sequence in sequences:
        #     for tokens in sequence:
        #         for token in tokens:
        #             add_definition_for(token)

        print('Adding definitions...')
        for key in tqdm(keys):
            add_definition_for(key)

        print('Add some text...')
        context.initial_weight = 0.2
        context.weight_increase = 0.1
        context.add_text(text)
        context.add_text('my favorite food is salad')
        context.add_text('I enjoy greek salad')
        context.add_text('I like my salad with tomatoes and cucumbers')

        print('Starting stimulating...')
        context.stimulate_sequence('food', skip_decrease=True)

        print('Start rendering...')
        if record:
            context.start_recording('output/tests/output.gif',
                                    title="Test Gif", consider_stimulus=True, fps=3, arrow_size=arrow_size)

        if record:
            context.stop_recording()
        else:
            context.render('output/tests/output.png',
                           consider_stimulus=consider_stimulus, figsize=(20, 20), arrow_size=arrow_size)

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
