import os
import networkx as nx
import torch
import json
import spacy
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

nlp = spacy.load("en_core_web_sm")

initial_weight = 0.1
weight_increase = 0.05
degrade_factor = 0.8
stimulus = 1
temp_decrease = 0.1
factor_constant = 0.75


def get_dictionary_keys(dictionary='dictionaries/essential.json'):
    dictionary = json.loads(
        open(dictionary).read())
    keys = list(dictionary.keys())
    keys.append('<start>')
    keys.append('<end>')

    return keys


def get_initial_graph(keys):
    G = nx.DiGraph()
    for key in keys:
        G.add_node(keys.index(key), s=0)

    G.add_node(keys.index('<start>'), s=0)
    G.add_node(keys.index('<end>'), s=0)

    return G


def stimulate(key, value, G, factor=degrade_factor, to_set=None, nodes=None):
    if key in G.nodes():
        value = float(value) * factor

        if to_set is None:
            to_set = {}
        if nodes is None:
            nodes = G.nodes()

        if key not in to_set:
            node = G[key]
            previous_value = nodes[key]['s']
            new_value = previous_value + value
            to_set[key] = {'s':  min(1, new_value)}

            if new_value > 0.1:
                for sub_key in node.keys():
                    if sub_key != key:
                        weight = node[sub_key]['weight']
                        stimulate(sub_key, weight * new_value,
                                  G, factor * factor_constant, to_set=to_set, nodes=nodes)

    return to_set


def do_link(from_tkn, to_tkn, G):
    from_tkn = from_tkn
    to_tkn = to_tkn
    if G.has_edge(from_tkn, to_tkn):
        G[from_tkn][to_tkn]['weight'] += weight_increase
    else:
        G.add_edge(from_tkn, to_tkn, weight=initial_weight)


def create_links(text, keys, current_graph):
    doc = nlp(text)
    for sentence in doc.sents:
        prev = keys.index('<start>')
        for token in sentence:
            if not token.is_punct:
                lower = token.lemma_.lower()
                if lower in keys:
                    lower_index = keys.index(lower)
                    do_link(prev, lower_index,
                            current_graph)
                    prev = lower_index
                else:
                    print(lower)
        do_link(prev, keys.index('<end>'), current_graph)

    return doc


class ContextualGraphDataset(InMemoryDataset):
    def __init__(self, source, transform=None, pre_transform=None, pre_filter=None):
        print('Initializing')
        self.source = source
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

    def process(self):

        keys = get_dictionary_keys()
        G = get_initial_graph(keys)

        data_list = []

        def add_to_list(data_graph, lower_lemma):
            nodes = data_graph.nodes(data=True)
            edges = data_graph.edges(data=True)

            y0 = [edge[0] for edge in edges]
            y1 = [edge[1] for edge in edges]
            weights = [edge[2]['weight'] for edge in edges]

            x = torch.tensor(
                [[node['s'] for idx, node in nodes]], dtype=torch.float)
            edge_index = torch.tensor([y0, y1], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=weights, y=torch.tensor([lower_lemma]))

            data_list.append(data)

        for root, dirs, files in os.walk(self.source):
            for file in files:
                current_graph = G.copy()
                path = os.path.join(root, file)

                with open(path) as f:
                    text = f.read()

                    doc = create_links(text, keys, current_graph)

                    data_graph = current_graph.copy()
                    for sentence in doc.sents:
                        prev = keys.index('<start>')
                        to_set = stimulate(prev, stimulus, data_graph)
                        for token in sentence:
                            if not token.is_punct and token.lemma_.lower() in keys:
                                lower_lemma = keys.index(token.lemma_.lower())
                                add_to_list(data_graph, lower_lemma)

                                to_set = stimulate(
                                    lower_lemma, stimulus, data_graph)
                                prev = lower_lemma
                                nx.set_node_attributes(data_graph, to_set)

                        add_to_list(data_graph, keys.index('<end>'))

                        ## Decrease temperature ##
                        dlist = data_graph.nodes(data=True)
                        set_values = {}
                        for node, data in dlist:
                            new_value = data['s'] - temp_decrease
                            set_values[node] = {'s': max(0, new_value)}
                        nx.set_node_attributes(data_graph, set_values)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
