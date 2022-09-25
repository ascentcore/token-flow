from itertools import count
import json
import networkx as nx
import numpy as np
from matplotlib.pyplot import figure

from .vocabulary import Vocabulary


class Context():

    render_label_size = 0.1

    def __init__(self, name,
                 vocabulary=None,
                 directed=True,
                 initial_weight=0.1,
                 definition_weight=0.1,
                 weight_increase=0.1,
                 neuron_opening=0.9,
                 default_stimulus=1,
                 max_stimulus=1,
                 propagate_threshold=0.1,
                 decay_if_low_signal=True,
                 temp_decrease=0.1):

        self.name = name

        self.plt = None

        # The default stimulus when a node is excited
        self.default_stimulus = default_stimulus

        # Initial weight when a connection is created between 2 tokens
        self.initial_weight = initial_weight

        # Setup the weight between a definition and a word
        self.definition_weight = definition_weight

        # Weight increase factor when a connection is already created
        self.weight_increase = weight_increase

        # Neuron opening reduce the stimulus when passing message
        self.neuron_opening = neuron_opening

        # If one node is stimulated with a value less than its stimulus it will not trigger and will decay stimulus
        self.decay_if_low_signal = decay_if_low_signal

        # Maximum value of the stimulus
        self.max_stimulus = max_stimulus

        # Stimuli decrease over time
        self.temp_decrease = temp_decrease

        # Threshold of stimulus to trigger a propagation
        self.propagate_threshold = propagate_threshold

        self.directed = directed

        self.graph = nx.DiGraph() if directed else nx.Graph()

        self.vocabulary = vocabulary

        self.vocabulary.register(self.on_new_words)

        # If the vocabulary is not empty then we can initialize the graph
        if self.vocabulary.size() > 0:
            self.initialize_nodes()

    def add_node(self, token):
        if not self.graph.has_node(token):
            self.graph.add_node(token, s=0)

    def on_new_words(self, words):
        # print(words)
        for token in words:
            self.add_node(token)

    def initialize_nodes(self):
        self.graph.add_nodes_from(self.vocabulary.vocabulary)
        # Set initial stimulus to 0
        nx.set_node_attributes(self.graph, 0, 's')

    def connect(self, from_token, to_token, weight=None):
        if from_token != to_token:
            if self.graph.has_edge(from_token, to_token):
                current_weight = self.graph[from_token][to_token]['weight']
                current_weight = min(1, current_weight + self.weight_increase)
                self.graph[from_token][to_token]['weight'] = current_weight
            else:
                self.graph.add_edge(from_token, to_token,
                                    weight=weight if weight != None else self.initial_weight)

    def from_sequence(self, sequences):
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                for from_token in sequence[i]:
                    for to_token in sequence[i + 1]:
                        self.connect(from_token, to_token)

    def add_text(self, text, skip_connections=False, decrease_on_end=None, append_to_vocab=True):
        _, sequences = self.vocabulary.add_text(
            text, append_to_vocab=append_to_vocab)

        if not skip_connections:
            self.from_sequence(sequences)
            if decrease_on_end != None:
                self.decrease_stimulus(decrease_on_end)

        return sequences

    def add_definition(self, word, definition, one_way=False, debug=False, append_to_vocab=True):
        if debug:
            print(f'Adding definition of {word}. with {definition}')
        missing, sequences = self.vocabulary.get_token_sequence(
            definition, append_to_vocab=append_to_vocab)

        self.vocabulary.add_to_vocabulary(word)
        self.add_node(word)

        for sequence in sequences:
            for i in range(0, len(sequence)):
                for tokens in sequence[i]:
                    self.connect(word, tokens, weight=self.definition_weight)

                    if not one_way:
                        self.connect(
                            tokens, word, weight=self.definition_weight)

        return missing, sequences

    def decrease_stimulus(self, decrease=None):
        if decrease is None:
            decrease = self.temp_decrease
        nodes = self.graph.nodes
        for node in self.graph.nodes():

            nodes[node]['s'] = max(0, nodes[node]['s'] - decrease)

    def prune_edges(self, threshold):
        long_edges = list(filter(
            lambda e: e[2] < threshold, (e for e in self.graph.edges.data('weight'))))
        le_ids = list(e[:2] for e in long_edges)

        # remove filtered edges from graph G
        self.graph.remove_edges_from(le_ids)

    def add_variations(self, text, variations):
        self.vocabulary.add_to_vocabulary(text)
        self.add_node(text)
        for variation in variations:
            self.vocabulary.add_to_vocabulary(variation)
            self.add_node(variation)
            self.connect(text, variation, 1)
            self.connect(variation, text, 1)

    def stimulate(self, token, stimulus=None, to_set=None, decrease_factor=None, skip_decrease=False, max_depth=10):
        if max_depth > 0:
            root = False
            if token in self.vocabulary.vocabulary:
                if to_set is None:
                    if not skip_decrease:
                        self.decrease_stimulus(decrease_factor)
                    to_set = {}
                    root = True

                if stimulus is None:
                    stimulus = self.default_stimulus

                graph = self.graph
                node = graph.nodes[token]
                current_stimulus = node['s']
                pass_message = False
                if token not in to_set.keys() or to_set[token]['s'] < stimulus:
                    if current_stimulus < stimulus:
                        current_stimulus = stimulus
                        pass_message = True
                    # elif self.decay_if_low_signal:
                    #     current_stimulus = graph.nodes[token]['s'] - \
                    #         self.temp_decrease

                    current_stimulus = max(0, min(1, current_stimulus))
                    to_set[token] = {'s':  current_stimulus}
                    if current_stimulus > self.propagate_threshold:
                        if pass_message:
                            node = graph[token]
                            for sub_key in node.keys():
                                if sub_key != token:
                                    weight = node[sub_key]['weight']
                                    self.stimulate(sub_key,
                                                   weight * current_stimulus * self.neuron_opening, to_set=to_set, max_depth=max_depth - 1)

            if root:
                nx.set_node_attributes(self.graph, to_set)

        return to_set

    def stimulate_sequence(self, sequence, stimulus=None, decrease_factor=None, skip_decrease=False):
        _, sentences = self.vocabulary.get_token_sequence(
            sequence, append_to_vocab=False)
        for sentence in sentences:
            for tokens in sentence:
                for token in tokens:
                    self.stimulate(
                        token, stimulus, decrease_factor=decrease_factor, skip_decrease=skip_decrease)

    def get_stimulus_of(self, token):
        return self.graph.nodes[token]['s']

    def get_stimuli(self):
        return [self.get_stimulus_of(token) for token in self.vocabulary.vocabulary]

    def get_top_stimuli(self, count=10):
        return sorted([(token, self.get_stimulus_of(token)) for token in self.vocabulary.vocabulary], key=lambda x: x[1], reverse=True)[:count]

    def get_matrix(self):
        return nx.to_numpy_matrix(self.graph)
        # return np.expand_dims(nx.to_numpy_matrix(self.graph), axis=0)

    def store(self, path):

        settings = {
            'directed': self.directed,
            'initial_weight': self.initial_weight,
            'weight_increase': self.weight_increase,
            'neuron_opening': self.neuron_opening,
            'default_stimulus': self.default_stimulus,
            'max_stimulus': self.max_stimulus,
            'propagate_threshold': self.propagate_threshold,
            'decay_if_low_signal': self.decay_if_low_signal,
            'temp_decrease': self.temp_decrease,
        }

        with open(f'{path}/{self.name}.settings.json', 'w') as outfile:
            outfile.write(json.dumps(settings, indent=2))

        nx.write_edgelist(self.graph, f'{path}/{self.name}.edgelist')

    @ classmethod
    def from_file(cls, path, name, vocabulary):

        settings = json.loads(open(f'{path}/{name}.settings.json').read())

        context = cls(name,
                      vocabulary,
                      directed=settings['directed'],
                      initial_weight=settings['initial_weight'],
                      weight_increase=settings['weight_increase'],
                      neuron_opening=settings['neuron_opening'],
                      default_stimulus=settings['default_stimulus'],
                      max_stimulus=settings['max_stimulus'],
                      propagate_threshold=settings['propagate_threshold'],
                      decay_if_low_signal=settings['decay_if_low_signal'],
                      temp_decrease=settings['temp_decrease'])

        context.initialize_nodes()

        if settings['directed']:
            edgelist_graph = nx.read_edgelist(
                f'{path}/{context.name}.edgelist', create_using=nx.DiGraph)
        else:
            edgelist_graph = nx.read_edgelist(
                f'{path}/{context.name}.edgelist')

        try:
            for edge in edgelist_graph.edges(data=True):
                context.graph.add_edge(
                    edge[0], edge[1], weight=edge[2]['weight'])
        except Exception as e:
            print('Error on context', context.name)
            print(f'Edge: {edge}')
            raise('Runtime error on edge addition')

        return context

    def load_text_file(self, path, append_to_vocab=False):
        with open(path, 'r') as f:
            text = f.read()
            self.add_text(text, append_to_vocab=append_to_vocab)

    def pretty_print(self, sorted=False):
        print('|  Stimulus  |               Token               |')
        for token, value in [(token, self.get_stimulus_of(token)) for token in self.vocabulary.vocabulary]if sorted == False else self.get_top_stimuli(count=len(self.vocabulary.vocabulary)):
            print(f'  {value:.2f}  | {token}')
        # return [self.get_stimulus_of(token) for token in self.vocabulary.vocabulary]

    def render(self, path, title="Graph Context", consider_stimulus=True, arrow_size=3, pre_pos=None, force_text_rendering=False, skip_empty_nodes=False, figsize=(14, 14), dpi=150, seed=12345):

        graph = self.graph

        if skip_empty_nodes:
            graph = graph.copy()
            graph.remove_nodes_from(list(nx.isolates(graph)))

        if self.plt is None:
            plt = figure(figsize=figsize, dpi=dpi)
        else:
            plt = self.plt

        if pre_pos:
            pos = pre_pos
        else:
            # pos = nx.spring_layout(self.graph, k=0.3, iterations=50)
            # pos = nx.kamada_kawai_layout(self.graph, weight='weight')
            pos = nx.fruchterman_reingold_layout(graph, seed=seed)
            # pos = nx.circular_layout(self.graph)

        edge_width = []
        for edge in graph.edges(data=True):
            edge_width.append(edge[2]['weight'] * 0.1)

        stimulus = nx.get_node_attributes(graph, 's')

        if consider_stimulus:
            labels = {n: n if stimulus[n] >
                      self.render_label_size or force_text_rendering else '' for n in graph}
        else:
            labels = {n: n for n in graph}

        node_size = [stimulus[n] *
                     300 if consider_stimulus else stimulus[n] * 100 for n in graph]

        ax = plt.gca()
        # ax.set_xlim([-1.4, 1.4])
        # ax.set_ylim([-1.4, 1.4])
        ax.set_title(title)

        nx.draw_networkx_nodes(
            graph, pos=pos, node_size=node_size, node_color='none', edgecolors='red', linewidths=0.3)
        nx.draw_networkx_labels(graph, pos, font_size=6, labels=labels,
                                verticalalignment='bottom')
        nx.draw_networkx_edges(
            graph, pos, width=edge_width, arrowsize=arrow_size)
        plt.tight_layout()
        plt.savefig(path)
        plt.clf()

        return pos
