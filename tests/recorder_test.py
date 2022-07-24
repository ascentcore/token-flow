from src.context import Context
from src.recorder import Recorder
from src.vocabulary import Vocabulary

import unittest


class RectorderTest(unittest.TestCase):

    def test_record(self):

        context = Recorder('recorder')
        context.add_text(
            "tomato is a round, red fruit with a lot of seeds, eaten cooked or uncooked as a vegetable, for example in salads or sauces", include_start=False)
        context.add_text(
            "vegetable is a plant, root, seed, or pod that is used as food, especially in dishes that are not sweet", include_start=False)
        context.add_text(
            "potato is a round vegetable that grows underground and has white flesh with light brown, red, or pink skin, or the plant on which these grow", include_start=False)
        # context.add_text("", include_start=False)

        _, sequences = context.vocabulary.get_token_sequence(
            "it tomato a vegetable", include_start=False)

        context.start_recording('output/tests/output.gif',
                                title="Test Gif", consider_stimulus=True, fps=5)

        for sequence in sequences:
            for token in sequence:
                context.stimulate(token)

        context.stop_recording()


if __name__ == '__main__':
    unittest.main()
