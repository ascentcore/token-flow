import os
import spacy
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from spacy.matcher import DependencyMatcher

from torch_geometric.data import Data


class Context:

    # Initial weight of the edge when created
    initial_weight = 0.1

    # Edge weight increase when connection exists
    weight_increase = 0.01

    # Degradation of signal from one node to the childre
    neuron_opening = 0.75

    # Initial stimulus
    stimulus = 1

    # maximum allowed stimulus
    max_stimulus = 1

    # Stimulus propagation threshold
    propagate_threshold = 0.1

    # Stimulus decrease
    temp_decrease = 0.1

    # Rendering only
    render_label_size = 0.01

    vocabulary = ['<start>', '<end>']

    def __init__(self, path = None):
        self.nlp = spacy.load("en_core_web_sm")
        if path:
            if os.path.exists(f'{path}/dataset/vocabulary.txt'):
                print('Vocabulary found, loading...')
                with open(f'{path}/dataset/vocabulary.txt', 'r') as fp:
                    self.vocabulary = []
                    for line in fp:
                        self.vocabulary.append(line[:-1])

                self.G = nx.DiGraph()
                for key in self.vocabulary:
                    self.G.add_node(self.vocabulary.index(key))

                nx.set_node_attributes(self.G, 0, 's')

    def from_folder(self, folder_path, reset=False, connect_all=True):
        graphs = {}
        if not os.path.exists(f'{folder_path}/dataset'):
            os.makedirs(f'{folder_path}/dataset')

        if os.path.exists(f'{folder_path}/dataset/vocabulary.txt') and not reset:
            print('Vocabulary found, loading...')
            with open(f'{folder_path}/dataset/vocabulary.txt', 'r') as fp:
                self.vocabulary = []
                for line in fp:
                    self.vocabulary.append(line[:-1])
            for file in os.listdir(folder_path):
                if file.endswith(".txt"):
                    G = nx.DiGraph()
                    for i in range(0, len(self.vocabulary)):
                        G.add_node(i)

                    edgelist_graph = nx.read_edgelist(
                        f'{folder_path}/dataset/{file}-edgelist.txt')
                    for edge in edgelist_graph.edges(data=True):
                        G.add_edge(int(edge[0]), int(
                            edge[1]), weight=edge[2]['weight'])

                    nx.set_node_attributes(G, 0, 's')
                    graphs[file] = G

        else:
            print('Vocabulary not found, creating...')
            for file in os.listdir(folder_path):
                if file.endswith(".txt"):
                    filename = os.path.join(folder_path, file)
                    with open(filename, 'r') as fp:
                        print('Parsing file: ' + filename)
                        text = fp.read()
                        G = self.from_text(
                            text, keys=self.vocabulary,
                            set_vocabulary=False,
                            set_graph=False,
                            connect_all=connect_all)

                        graphs[file] = G

                        if not os.path.exists(f'{folder_path}/dataset/{file}-edgelist.txt') or reset:
                            print('Creating edgelist...')
                            nx.write_edgelist(
                                G, f'{folder_path}/dataset/{file}-edgelist.txt')
                                

            with open(f'{folder_path}/dataset/vocabulary.txt', 'w') as fp:
                for token in self.vocabulary:
                    fp.write(token + '\n')

            for G in graphs.values():
                for key in self.vocabulary:
                    if key not in G.nodes():
                        G.add_node(self.vocabulary.index(key))

                nx.set_node_attributes(G, 0, 's')

        return graphs

    def from_text(self, text, all_tokens=True, accepted=['NOUN', 'PROPN', 'VERB'], keys=['<start>', '<end>'], set_vocabulary=True, set_graph=True, connect_all=True):
        G = nx.DiGraph()
        doc = self.nlp(text)

        def connect_tokens(tokens):
            for i in range(0, len(tokens)):
                for j in range(i+1, len(tokens)):

                    # do not link start with end and make sure that start points just to the first node
                    if tokens[i] == 0 and (tokens[j] == 1 or j > 1):
                        continue

                    if connect_all or j == i + 1 or tokens[i] != tokens[j]:
                        if tokens[i] != tokens[j]:
                            if G.has_edge(tokens[i], tokens[j]):
                                weight = G[tokens[i]][tokens[j]]['weight']
                                G[tokens[i]][tokens[j]]['weight'] = min(
                                    weight + self.weight_increase, 1)
                            else:
                                G.add_edge(tokens[i], tokens[j],
                                           weight=self.initial_weight)

        for sentence in doc.sents:
            tokens = [0]
            for token in sentence:
                if not token.is_punct and token.text != '\n':
                    token_lower = token.lemma_.lower()

                    if all_tokens or token.pos_ in accepted:
                        if token_lower not in keys:
                            keys.append(token_lower)
                        tokens.append(keys.index(token_lower))

                    if token.text.lower() not in keys:
                        keys.append(token.text.lower())
                else:
                    tokens.append(1)
                    connect_tokens(tokens)
                    tokens = [0]

            if len(tokens) > 0:
                connect_tokens(tokens)

        nx.set_node_attributes(G, 0, 's')

        if set_vocabulary:
            self.vocabulary = keys

        if set_graph:
            self.G = G

        return G

    def initialize_from_edgelist(self, path):
       edgelist_graph = nx.read_edgelist(path)
       output_graph = self.G.copy()
       for edge in edgelist_graph.edges(data=True):
            output_graph.add_edge(int(edge[0]), int(
                edge[1]), weight=edge[2]['weight'])

       return output_graph

    def translate(self, token_index):
        return self.vocabulary[token_index]

    def get_tensor_from_nodes(self, data_graph, next_token=None):
        nodes = data_graph.nodes(data=True)
        edges = data_graph.edges(data=True)

        n_nodes = len(self.vocabulary)

        y0 = [edge[0] for edge in edges]
        y1 = [edge[1] for edge in edges]
        weights = [[edge[2]['weight']] for edge in edges]

        x = torch.tensor(
            [[node['s']] for idx, node in nodes], dtype=torch.float)

        edge_index = torch.tensor([y0, y1], dtype=torch.long)

        if next_token is not None:
            next_token_index = self.get_token_index(next_token)
            y = torch.zeros(n_nodes, dtype=torch.long)
            y[next_token_index] = 1
            data = Data(x=x, edge_index=edge_index, edge_attr=weights, y=y)
        else:
            data = Data(x=x, edge_index=edge_index, edge_attr=weights)

        data.num_nodes = n_nodes

        return data

    def get_token_index(self, token):
        return self.vocabulary.index(token)

    def create_links(self, graph, token_list):
        previous = None
        for token in token_list:
            token_index = self.get_token_index(token)
            if previous is not None:
                if graph.has_edge(previous, token_index):
                    weight = graph[previous][token_index]['weight']
                    graph[previous][token_index]['weight'] = min(
                        weight + self.weight_increase, self.max_stimulus)
                else:
                    graph.add_edge(previous, token_index,
                                   weight=self.initial_weight)
            previous = token_index

    def stimulate_token(self, graph, token, stimulus=stimulus, debug=False):
        token_index = self.get_token_index(token)
        to_set = self.stimulate(graph, token_index, stimulus, {})
        if debug:
            print(token, '->', [(self.vocabulary[idx], str(round(to_set[idx]['s'], 3)))
                                for idx in to_set.keys()])
        nx.set_node_attributes(graph, to_set)

    def decrease_stimulus(self, graph, decrease=temp_decrease):
        for node in graph.nodes():
            graph.nodes[node]['s'] = max(0, graph.nodes[node]['s'] - decrease)

    def get_token_index(self, token):
        return self.vocabulary.index(token)

    def stimulate(self, graph, token_index, stimulus=stimulus, to_set=None, and_set=False):
        if to_set is None:
            to_set = {}

        if token_index not in to_set.keys():
            node = graph[token_index]
            print(self.vocabulary)
            print(token_index, self.vocabulary[token_index])
            if graph.nodes[token_index]['s'] < stimulus:
                # current_stimulus = graph.nodes[token_index]['s'] + stimulus
                current_stimulus = stimulus
            else:
                current_stimulus = graph.nodes[token_index]['s'] - \
                    self.temp_decrease

            to_set[token_index] = {'s': max(0, min(1, current_stimulus))}

            if stimulus > self.propagate_threshold:
                for sub_key in node.keys():

                    if sub_key != token_index:
                        weight = node[sub_key]['weight']
                        self.stimulate(graph, sub_key,
                                       weight * stimulus * self.neuron_opening, to_set=to_set)

        if and_set:
            nx.set_node_attributes(graph, to_set)

        return to_set

    plt = figure(figsize=(8, 6), dpi=150)

    def render(self, path, title="Context", consider_stimulus=True, pre_pos=None, arrowsize=3):

        if pre_pos:
            pos = pre_pos
        else:
            pos = nx.kamada_kawai_layout(self.G, scale=2)

        edge_width = []
        for edge in self.G.edges(data=True):
            edge_width.append(edge[2]['weight'] * 0.1)

        stimulus = nx.get_node_attributes(self.G, 's')

        if consider_stimulus:
            labels = {n: self.translate(n) if stimulus[n] >
                      self.render_label_size else '' for n in self.G}
        else:
            labels = {n: self.translate(n) for n in self.G}

        node_size = [stimulus[n] *
                     300 if consider_stimulus else 50 for n in self.G]

        ax = plt.gca()
        # ax.set_xlim([-1.4, 1.4])
        # ax.set_ylim([-1.4, 1.4])
        ax.set_title(title)

        nx.draw_networkx_nodes(
            self.G, pos=pos, node_size=node_size, node_color='none', edgecolors='red', linewidths=0.3)
        nx.draw_networkx_labels(self.G, pos, font_size=6, labels=labels,
                                verticalalignment='bottom')
        nx.draw_networkx_edges(
            self.G, pos, width=edge_width, arrowsize=arrowsize)
        plt.tight_layout()
        plt.savefig(path)
        plt.clf()

        return pos
