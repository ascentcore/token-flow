import networkx as nx
from matplotlib.pyplot import figure

from .vocabulary import Vocabulary
from .context import Context


class Dataset():

    def __init__(self):
        self.vocabulary = Vocabulary()
        self.context_data = {}

    def add_text(self, context_name, text):
        if context_name not in self.context_data.keys():
            context = Context(context_name, vocabulary=self.vocabulary)
            self.context_data[context_name] = {
                "context": context,
                "sentences": []
            }

        current_context = self.context_data[context_name]
        sents = current_context["context"].add_text(text)
        current_context["sentences"] += sents

    def get_dataset(self):
        dataset = []
        for context_name in self.context_data.keys():
            context_data = self.context_data[context_name]

            context = context_data["context"]
            graph_matrix = context.get_matrix()
            sentences = context_data["sentences"]
            for sent in sentences:
                for token in sent:
                    x = context.get_stimuli()
                    context.stimulate(token)
                    y = context.get_stimuli()
                    dataset.append((graph_matrix, x, y))

        return dataset

