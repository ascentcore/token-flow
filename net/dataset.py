import os
import torch
import spacy
import networkx as nx
from torch_geometric.data import InMemoryDataset


from net.context import Context

nlp = spacy.load("en_core_web_sm")


class ContextualGraphDataset(InMemoryDataset):

    def get_dictionary_keys(self, dictionary=None):
        keys = []
        with open(self.get_dictionary_path(dictionary), 'r') as fp:
            for line in fp:
                x = line[:-1]
                keys.append(x)

        return keys

    def get_dictionary_path(self, dictionary=None):
        return f'{self.source}/dataset/dictionary.txt' if dictionary is None else dictionary

    def __init__(self, source, prune_dictionary=False, transform=None, pre_transform=None, pre_filter=None):
        print('Initializing')
        self.source = source
        self.prune_dictionary = prune_dictionary

        super().__init__(f'{source}/dataset',
                         transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dataset.pt']

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        pass

    def get_sentence_tokens(self, text, keys):
        sentences = []
        doc = nlp(text)
        for sentence in doc.sents:
            tokens = ['<start>']
            for token in sentence:
                if not token.is_punct:
                    lemma = token.lemma_.lower()
                    if lemma in keys:
                        tokens.append(lemma)

            tokens.append('<end>')
            sentences.append(tokens)

        return sentences

    def process(self):

        print(f'Starting processing ...')

        texts = []

        all_keys = self.get_dictionary_keys('dictionaries/essential.txt')
        used_keys = ['<start>', '<end>']

        for root, dirs, files in os.walk(self.source):
            for file in files:

                path = os.path.join(root, file)
                with open(path) as f:
                    basename = os.path.basename(f.name)
                    print(f'Processing {basename}')
                    try:
                        text = f.read()
                        content = {
                            'filename': basename,
                            'sentences': []
                        }
                        doc = nlp(text)
                        for sentence in doc.sents:
                            tokens = ['<start>']
                            for token in sentence:
                                if not token.is_punct:
                                    lemma = token.lemma_.lower()
                                    if lemma in all_keys:
                                        tokens.append(lemma)

                                        if self.prune_dictionary and lemma not in used_keys:
                                            used_keys.append(lemma)

                            tokens.append('<end>')

                            content['sentences'].append(tokens)

                        texts.append(content)
                    except:
                        print(f'Error processing {basename}')
                        pass

            break

        print(f'Storing dictionary to {self.get_dictionary_path()} ...')
        used_keys.append('')
        with open(self.get_dictionary_path(), 'w') as fp:
            fp.write('\n'.join(used_keys if self.prune_dictionary else all_keys))

        #### Start creating context graphs ####
        context = Context(self.get_dictionary_path())

        data_list = []
        for txt in texts:
            graph = context.G.copy()
            name = txt['filename']
            sentences = txt['sentences']

            for sentence in sentences:
                context.create_links(graph, sentence)

            nx.write_edgelist(
                graph, f'{self.source}/dataset/{name}-graph.edgelist')

            for sentence in sentences:
                for token in sentence:
                    data_list.append(
                        context.get_tensor_from_nodes(graph, token))
                    if token == '<end>':

                        context.decrease_stimulus(graph)
                    else:
                        context.stimulate_token(graph, token)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
