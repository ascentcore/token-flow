import os
import torch
import spacy
from tqdm import tqdm
import networkx as nx
from torch_geometric.data import InMemoryDataset


from net.context import Context

nlp = spacy.load("en_core_web_sm")


class ContextualGraphDataset(InMemoryDataset):

    def __init__(self, source, transform=None, pre_transform=None, pre_filter=None, from_scratch=False):

        if from_scratch:
            import shutil
            try:
                shutil.rmtree(f'{source}/dataset')
            except:
                pass

        print('Initializing')
        self.source = source
       
        super().__init__(f'{source}/dataset',
                         transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_vocabulary_keys(self, vocabulary=None):
        keys = []
        with open(self.get_vocabulary_path(vocabulary), 'r') as fp:
            for line in fp:
                x = line[:-1]
                keys.append(x)

        return keys

    def get_vocabulary_path(self, vocabulary=None):
        return f'{self.source}/dataset/vocabulary.txt' if vocabulary is None else vocabulary

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
            tokens = [('<start>', '<start>')]
            for token in sentence:
                if not token.is_punct:
                    lemma = token.lemma_.lower()
                    if lemma in keys:
                        tokens.append((lemma, token.text.lower()))

            tokens.append(('<end>', '<end>'))
            sentences.append(tokens)

        return sentences

    def process(self):
        print('Processing dataset ...')
        texts = []

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
                            tokens = [('<start>', '<start>')]
                            for token in sentence:

                                # TODO: check if _is_punct is correct
                                if not token.is_punct and token.text != '\n':
                                    tokens.append((token.lemma_.lower(), token.text.lower()))
                            tokens.append(('<end>', '<end>'))
                            content['sentences'].append(tokens)
                        texts.append(content)
                    except:
                        print(f'Error processing {basename}')
                        pass

            break

        context = Context()
        graphs = context.from_folder(f'{self.source}', connect_all=False)
        
        data_list = []
        for txt in texts:

            name = txt['filename']
            sentences = txt['sentences']
            graph = graphs[name]
            context.G = graph
            context.render(f'./output/sample-{name}.jpg', consider_stimulus=False)
            for sentence in tqdm(sentences, 'Stimulating nodes'):
                for token in sentence:
                    data_list.append(
                        context.get_tensor_from_nodes(graph, token))
                    # context.stimulate_token(graph, token[0], debug=False)
                    context.stimulate_token(graph, token[1], debug=False)
                context.decrease_stimulus(graph, 0.1)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
