import re
import os
import json
import shutil
import numpy as np
from tqdm import tqdm

from .vocabulary import Vocabulary
from .context import Context


class BasicInMemDataset():

    def __init__(self, context):
        self.name = context.name
        self.context = context
        self.data = []

    def add_text(self, text, stimulus=None, decrease_on_end=None):
        _, sentences = self.context.vocabulary.get_token_sequence(text)
        input = self.context.get_stimuli()
        for sentence in sentences:
            for tokens in sentence:
                for token in tokens:
                    self.context.stimulate(token, stimulus)
                    output = self.context.get_stimuli()
                    filtered_output = [1 if x == 1 else 0 for x in output]
                    self.data.append((input, filtered_output))
                    input = output
            if decrease_on_end != None:
                self.context.decrease_stimulus(decrease_on_end)
                input = self.context.get_stimuli()


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

    def delete_context(self, context):
        del self.contexts[context]

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

    def reset_stimulus(self):
        for context in self.contexts.values():
            context.decrease_stimulus(1)

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

    def from_folder(self, folder, get_context, decrease_on_end=None):
        res = []
        for (dir_path, dir_names, file_names) in os.walk(folder):
            res.extend(file_names)

        for file_name in res:
            if not self.has_context(file_name):
                context = get_context(file_name)
                self.add_context(context)
            else:
                context = self.get_context(file_name)

            file = open(f'{folder}/{file_name}', 'r')
            for line in tqdm(file.readlines()):
                line = line.strip().lower()
                context.add_text(line, decrease_on_end)

        for file_name in res:
            dataset = self.get_dataset(file_name, file_name)
            file = open(f'{folder}/{file_name}', 'r')
            for line in tqdm(file.readlines()):
                line = line.strip().lower()
                dataset.add_text(line, decrease_on_end)

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

    @classmethod
    def get_sample_data(cls, context, text, stimulus=None):
        _, sentences = context.vocabulary.get_token_sequence(
            text, append_to_vocab=False)
        input = context.get_stimuli()
        input_data = [input]
        for sentence in sentences:
            for tokens in sentence:
                for token in tokens:
                    context.stimulate(token, stimulus)
                    input_data.append(context.get_stimuli())

        return input_data
