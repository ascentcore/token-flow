import os
import spacy
import torch
import networkx as nx
import collections
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from spacy.matcher import DependencyMatcher

from torch_geometric.data import Data


class Context:

    # Initial weight of the edge when created
    initial_weight = 0.1

    # Edge weight increase when connection exists
    weight_increase = 0.05

    # Degradation of signal from one node to the children
    neuron_opening = 0.5

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

    plt = None

    def __init__(self, path=None):
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

    def get_lemma(self, token, accepted=['NOUN', 'PROPN', 'VERB']):
        tokens = self.nlp(token)
        val = token
        if (tokens[0].pos_ in accepted):
            if tokens[0].lemma_ in self.vocabulary:
                val = tokens[0].lemma_

        return val

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

    def from_text(self,
                  text,
                  accepted=['NOUN', 'PROPN', 'VERB'],
                  keys=['<start>', '<end>'],
                  set_vocabulary=True,
                  set_graph=True,
                  connect_all=True):

        ## The graph contains only indexes of tokens
        ##       [0]
        ##      /   \
        ##    [2]    [4]
        ##   /   \  /
        ## [3]----[1]
        ##
        ##                                   [0]     [1]   [2]  [3]  [4]
        # Vocabulary in this case can be [<start>, <end>, 'a', 'b', 'c']

        G = nx.DiGraph()
        doc = self.nlp(text)

        def make_connection(from_nodes, to_node):
            for from_node in from_nodes:
                if from_node != to_node:
                    if G.has_edge(from_node, to_node):
                        weight = G[from_node][to_node]['weight']
                        G[from_node][to_node]['weight'] = min(
                            weight + self.weight_increase, 1)
                    else:
                        G.add_edge(from_node, to_node,
                                   weight=self.initial_weight)

        def connect_tokens(tokens):
            for i in range(0, len(tokens)):
                for j in range(i+1, len(tokens)):
                    if tokens[i] == 0 and (tokens[j] == 1 or j > 1):
                        continue

                    if connect_all or j == i + 1 or tokens[i] != tokens[j]:
                        if tokens[i] != tokens[j]:
                            make_connection([tokens[i]], tokens[j])
                            make_connection([tokens[j]], tokens[i])

        for sentence in doc.sents:
            tokens = []
            previous = [0]
            for token in sentence:
                current_position = []
                if token.text != '\n':
                    if not token.is_punct:
                        token_lower = token.lemma_.lower().strip()
                        text_lower = token.text.lower().strip()

                        if len(text_lower) > 0 and len(text_lower) > 0:

                            if token.pos_ in accepted:
                                # Add lemma to Graph and vocabulary.
                                # Lemma is the simplified word ('children' -> 'child')
                                token_index = len(keys)
                                if token_lower not in keys:
                                    G.add_node(token_index)
                                    keys.append(token_lower)

                                tokens.append(token_index)
                                make_connection(previous, token_index)

                                # The below lines allows connectivity to both word and lemma
                                # in 13. Jul we decided to remove it
                                # just to preserve the sentence structure
                                # this was to make a connection to the lemma as well
                                # we are going to keep it hidden for the moment
                                # current_sentence.append(token_index)

                            # If the actual word (not lemma) is not in the vocabulary add it (child might be but children not)
                            if text_lower not in keys:
                                G.add_node(len(keys))
                                keys.append(text_lower)

                            text_lower_index = keys.index(text_lower)

                            # This is strange - need to investigate
                            make_connection(previous, text_lower_index)

                            current_position.append(text_lower_index)

                            if len(current_position) > 1:
                                make_connection(
                                    [current_position[0]], current_position[1])
                                make_connection(
                                    [current_position[1]], current_position[0])

                            else:

                                previous = [0]

                            previous = current_position

                    elif len(tokens) > 0:
                        connect_tokens(tokens)
                        tokens = []

            make_connection(previous, 1)

        if len(tokens) > 0:
            connect_tokens(tokens)
            tokens = []

        nx.set_node_attributes(G, 0, 's')

        if G.has_edge(0, 1):
            G.remove_edge(0, 1)

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

    # Generate a pytorch tensor from a graph
    # If next token  (as word) is present then the data will include the y value
    # [data.x] represents the simulus of each node
    # [data.edge_index] contain all the edges in the graph
    # [data.edge_attr] contains the weight of each edge
    # [?data.y] an array of zeros with the same length as the number of tokens in vocabulary
    # have 1 as value to the element index of the next word
    ##
    # To analize if y should be a vector of the next 10 words??
    def get_tensor_from_nodes(self, data_graph, next_token=None):
        nodes = data_graph.nodes(data=True)
        edges = data_graph.edges(data=True)

        nnodes = {}
        for node in nodes:
            nnodes[node[0]] = node[1]['s']
        nodes = collections.OrderedDict(sorted(nnodes.items()))

        n_nodes = len(self.vocabulary)

        y0 = [edge[0] for edge in edges]
        y1 = [edge[1] for edge in edges]
        weights = [edge[2]['weight'] for edge in edges]

        x = torch.tensor(
            [[node[1]] for node in nodes.items()], dtype=torch.float)

        edge_index = torch.tensor([y0, y1], dtype=torch.long)

        if next_token is not None:
            next_token_index = self.get_token_index(next_token[1])
            y = torch.zeros(n_nodes, dtype=torch.long)
            # y = torch.tensor(
            # [node['s'] for idx, node in nodes], dtype=torch.float)
            #

            y[next_token_index] = self.stimulus
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
            # if graph.nodes[token_index]['s'] < stimulus:
            current_stimulus = graph.nodes[token_index]['s'] + stimulus
            # current_stimulus = stimulus
            # else:
            #     current_stimulus = graph.nodes[token_index]['s'] - \
            #         self.temp_decrease

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

    def render(self, path, title="Context", consider_stimulus=True, pre_pos=None, arrowsize=3):
        if self.plt is None:
            plt = figure(figsize=(3, 3), dpi=150)
        else:
            plt = self.plt

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
                     300 if consider_stimulus else 10 for n in self.G]

        ax = plt.gca()
        # ax.set_xlim([-1.4, 1.4])
        # ax.set_ylim([-1.4, 1.4])
        # ax.set_title(title)

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
