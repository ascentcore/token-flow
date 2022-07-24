from src.recorder import Recorder
from src.context import Context
from src.vocabulary import Vocabulary

import unittest
import json

"""
dictionary = open('assets/english_dictionary.txt', 'r')
# lines = dictionary.readlines()

# defs = []
# definition = []
# for line in lines:
#     res = line.strip()
#     if (res == ''):                
#         defs.append(definition)
#         definition = []
#     else:              
#         if len(definition) == 0:
#             print(res)  
#         definition.append(res)

# with open('assets/english_dictionary.json', 'w') as outfile:
#     outfile.write(json.dumps(defs))
"""


class TestContext(unittest.TestCase):

    def test_vocabulary_showcase(self):
        context = Context('Showcase', include_start=False, initial_weight=0.1)

        context.add_definition(
            'salad', 'The salad is a food made from a variety of fresh vegetables.')
        context.add_definition(
            'sausage', 'The sausage is a food meat product made from the meat of a pig.')
        
        context.add_definition(
            'car', 'A car is a wheeled motor vehicle used for transportation.')
        context.add_definition(
            'drive', 'To drive is to move the vehicle forward.')

        context.add_definition('eat', 'To eat is to consume food.')

        context.stimulate('drive', skip_decrease=True)
        context.stimulate('eat', skip_decrease=True)
        context.stimulate('food', skip_decrease=True)

        context.render('output/tests/output.png',
                       title="Definitions Only", consider_stimulus=True)

    def test_complex_showcase(self):

        context = Recorder('Showcase', include_start=False, initial_weight=0.2)
        context.add_definition(
            'tomato', 'The tomato is the edible berry commonly known as the tomato plant.')
        context.add_definition(
            'potato', 'The potato is a starchy tuber of the root of the potato plant.')
        context.add_definition(
            'cucumber', 'The cucumber is a widely cultivated plant in the gourd family, with an edible part.')
        context.add_definition(
            'avocado', 'The avocado is a tree-like fruit, botanically a berry')
        context.add_definition(
            'lettuce', 'The lettuce is a leafy green plant in the cabbage family.')
        context.add_definition(
            'spinach', 'The spinach is a vegetable, usually green, that is a member of the wintergreen family.')
        context.add_definition(
            'garlic', 'The garlic is a bulbous, bulbous, woody, bulb-like, perennial plant.')
        context.add_definition(
            'cheese', 'The cheese is a food produced by the milk of a cow.')
        context.add_definition(
            'salad', 'The salad is a food made from a variety of fresh vegetables.')

        _, sequences = context.vocabulary.get_token_sequence(
            "A Caesar salad is a green salad of romaine lettuce and croutons dressed with lemon juice, olive oil, egg, Worcestershire sauce, anchovies, garlic, Dijon mustard, Parmesan cheese, and black pepper.", include_start=False)

        context.start_recording('output/tests/output.gif',
                                title="Test Gif", consider_stimulus=True, fps=5, arrow_size=0.5)

        for sequence in sequences:
            for token in sequence:
                context.stimulate(token)

        for i in range(0, 10):
            context.decrease_stimulus()
            context.capture_frame()

        context.stop_recording()

    def test_vocabulary_initialization(self):
        context = Recorder('english_vocabulary',
                           initial_weight=0.1, include_start=False)

        dictionary = json.loads(
            open('assets/english_dictionary.json', 'r').read())

        context.vocabulary.get_token_sequence(
            'An engine or motor is a machine designed to convert one or more forms of energy into mechanical energy', append_to_vocab=True)

        for word in context.vocabulary.vocabulary.copy():
            print('Processing word: ' + word)
            if len(word) > 3:
                filtered_dict = [x for x in dictionary if x[0] == word]
                for definition in filtered_dict:
                    _, sequences = context.vocabulary.add_definition(
                        definition[0], definition[1])
                    context.from_sequence(sequences=sequences)

        context.store('.')
        _, sequences = context.vocabulary.get_token_sequence(
            "Mechanical heat engines convert heat into work via various thermodynamic processes. The internal combustion engine is perhaps the most common example of a mechanical heat engine, in which heat from the combustion of a fuel causes rapid pressurisation of the gaseous combustion products in the combustion chamber, causing them to expand and drive a piston, which turns a crankshaft.", include_start=False)

        context.start_recording('output/tests/output.gif',
                                title="Test Gif", consider_stimulus=True, fps=5, arrow_size=0.5)

        for sequence in sequences:
            for token in sequence:
                context.stimulate(token)

        for i in range(0, 10):
            context.decrease_stimulus()
            context.capture_frame()

        context.stop_recording()


if __name__ == '__main__':
    unittest.main()
