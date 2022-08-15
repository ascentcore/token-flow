from src.context.recorder import Recorder
from src.context.context import Context
from src.context.vocabulary import Vocabulary

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
        context = Recorder('Showcase', include_start=False, initial_weight=0.1)
        
        context.add_definition(
            'salad', 'a mixture of cold vegetables such as lettuce, tomato, and cucumber, served with a dressing.')

        context.add_definition(
            'lettuce', 'a plant with large, crisp leaves that can be eaten.'
        )

        context.add_definition('tomato', 'a red or yellow fruit with a juicy pulp. A tomato is eaten either raw or cooked as a vegetable.')
        context.add_definition('cucumber', 'a plant with a green, round, fleshy fruit that is used in salads, soups, and other dishes.')
        context.add_definition('vegetbale', 'a plant or part of a plant, such as carrots, beans, or lettuce, that is used for food.')

        context.add_definition('sausage', 'a mixture of chopped meat and spices stuffed into a casing of animal intestine.')

        context.add_definition('food', 'the flesh of animals when used as food.')


        context.add_definition('favorite', 'food is salad')
        context.add_definition('greek', 'salad olive oil, tomatoes, lettuce, and cucumber')
        
        context.add_text('my favorite food is salad')
        context.add_text('I like greek salad')
        context.add_text('my prefference in food are vegetables')
        context.add_text('I do not eat meat')

        print(context.vocabulary.vocabulary)

        context.start_recording('output/tests/output.gif',
                                title="Test Gif", consider_stimulus=True, fps=3, arrow_size=0.5)
                            

        context.stimulate_sequence('My favorite food is salad. I like vegetables in my salad. Like, lettuce, cucumber and tomatoes')
        
        context.render('output/tests/output.png', consider_stimulus=True)

        context.stop_recording()

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


if __name__ == '__main__':
    unittest.main()
