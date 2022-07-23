import networkx as nx
from matplotlib.pyplot import figure

from .vocabulary import Vocabulary
from .context import Context


class Dataset():

    def __init__(self):
        self.vocabulary = Vocabulary()
        self.contexts = {}

    def add_text(self, context_name, text):
        if context_name not in self.contexts.keys():
            context = Context(context_name, vocabulary=self.vocabulary)
            self.contexts[context_name] = context

        context = self.contexts[context_name]
        context.add_text(text)
