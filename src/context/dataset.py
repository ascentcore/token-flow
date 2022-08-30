import re
import os
import json
import shutil
import numpy as np

from .vocabulary import Vocabulary
from .context import Context


class BasicInMemDataset():

    def __init__(self, context):
        self.name = context.name
        self.context = context
        self.data = []

    def add_text(self, text, stimulus=None):
        _, sentences = self.context.vocabulary.get_token_sequence(text)
        input = self.context.get_stimuli()
        for sentence in sentences:
            for tokens in sentence:
                for token in tokens:
                    self.context.stimulate(token, stimulus)
                    output = self.context.get_stimuli()
                    self.data.append((input, output))
                    input = output


class Dataset():

    def __init__(self, vocabulary=None, default_context=None):
        self.vocabulary = vocabulary if vocabulary != None else Vocabulary()

        self.contexts = {
            'default': default_context if default_context != None else Context('default', self.vocabulary)
        }

        self.settings = {
            'contexts': ['default']
        }

        self.datasets = {

        }

    def add_context(self, context):
        self.contexts[context.name] = context
        self.settings["contexts"].append(context.name)

    def add_text(self, text):
        for context in self.contexts.values():
            context.add_text(text)

    def get_context(self, context_name='default'):
        return self.contexts[context_name]

    def has_context(self, context_name):
        return context_name in self.contexts.keys()

    def get_dataset(self, context_name='default', dataset_name='default'):
        context = self.contexts[context_name]

        if dataset_name in self.datasets.keys():
            return self.datasets[dataset_name]
        else:
            self.datasets[dataset_name] = BasicInMemDataset(context)
            return self.datasets[dataset_name]

    def store(self, path):
        try:
            try:
                shutil.rmtree(path)
            except:
                pass

            os.mkdir(path)
            self.vocabulary.save_vocabulary(path, 'vocabulary.json')
            for context in self.contexts.values():
                context.store(path)

            print(self.contexts.keys())
            print(self.datasets.keys())

            for dataset in self.datasets.values():
                with open(f'{path}/{dataset.name}.dataset.json', 'w') as outfile:
                    outfile.write(json.dumps(dataset.data))

            with open(f'{path}/dataset.settings.json', 'w') as outfile:
                outfile.write(json.dumps(self.settings))

        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    @classmethod
    def load(cls, path):
        vocabulary = Vocabulary.from_file(
            path, 'vocabulary.json')

        dataset = cls(vocabulary)

        settings = json.loads(
            open(f'{path}/dataset.settings.json').read())

        for context_name in settings['contexts']:
            if context_name != 'default':
                context = Context.from_file(
                    path, context_name, vocabulary)
                dataset.add_context(context)

        return dataset
