import networkx as nx
from matplotlib.pyplot import figure

from .vocabulary import Vocabulary


class Context():

    render_label_size = 0.01

    def __init__(self, name, directed=True, vocabulary=None,
                 initial_weight=0.1,
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

        self.graph = nx.DiGraph() if directed else nx.Graph()

        self.vocabulary = vocabulary if vocabulary is not None else Vocabulary()
        self.vocabulary.register(self.on_new_words)

        # If the vocabulary is not empty then we can initialize the graph
        if self.vocabulary.size() > 0:
            self.initialize_nodes()

    def on_new_words(self, words):
        print(self.name, 'new words:', words)
        for token in words:
            self.graph.add_node(token, s=0)

    def initialize_nodes(self):
        self.graph.add_nodes_from(self.vocabulary.vocabulary)
        # Set initial stimulus to 0
        nx.set_node_attributes(self.graph, 0, 's')

    def connect(self, from_token, to_token):
        if self.graph.has_edge(from_token, to_token):
            self.graph[from_token][to_token]['weight'] += self.weight_increase
        else:
            self.graph.add_edge(from_token, to_token,
                                weight=self.initial_weight)

    def add_text(self, text):
        _, sequences = self.vocabulary.add_text(text)

        for sequence in sequences:
            for i in range(len(sequence) - 1):
                self.connect(sequence[i], sequence[i + 1])

    def decrease_stimulus(self, decrease=None):
        if decrease is None:
            decrease = self.temp_decrease
        nodes = self.graph.nodes
        for node in self.graph.nodes():
            nodes[node]['s'] = max(0, nodes[node]['s'] - decrease)

    def get_matrix(self):
        return nx.to_numpy_matrix(self.graph)

    def stimulate(self, token, stimulus=None, to_set=None, decrease_factor=None):
        root = False
        if token in self.vocabulary.vocabulary:
            if to_set is None:
                self.decrease_stimulus(decrease_factor)
                to_set = {}
                root = True

            if stimulus is None:
                stimulus = self.default_stimulus

            graph = self.graph
            node = graph.nodes[token]
            print(token, node)
            current_stimulus = node['s']
            pass_message = False

            if token not in to_set.keys():
                if current_stimulus < stimulus:
                    current_stimulus = stimulus
                    pass_message = True
                elif self.decay_if_low_signal:
                    current_stimulus = graph.nodes[token]['s'] - \
                        self.temp_decrease

                current_stimulus = max(0, min(1, current_stimulus))
                to_set[token] = {'s':  current_stimulus}

                if pass_message:
                    node = graph[token]
                    for sub_key in node.keys():
                        if sub_key != token:
                            weight = node[sub_key]['weight']
                            self.stimulate(sub_key,
                                           weight * current_stimulus * self.neuron_opening, to_set=to_set)

        if root:
            nx.set_node_attributes(self.graph, to_set)

        return to_set

    def get_stimulus_of(self, token):
        return self.graph.nodes[token]['s']

    def get_stimuli(self):
        return self.graph.nodes(data='s')

    def store(self, path):
        nx.write_edgelist(self.graph, f'{path}/{self.name}.edgelist')

    def load(self, path):
        if self.vocabulary.size() == 0:
            raise Exception(
                'Vocabulary is empty. Please add some tokens before loading a context.')

        edgelist_graph = nx.read_edgelist(
            f'{path}/dataset/{self.name}.edgelist')
        for edge in edgelist_graph.edges(data=True):
            self.graph.add_edge(int(edge[0]), int(
                edge[1]), weight=edge[2]['weight'])

    def render(self, path, title="Graph Context", consider_stimulus=True, arrow_size=3, pre_pos=None, force_text_rendering = False):
        if self.plt is None:
            plt = figure(figsize=(3, 3), dpi=150)
        else:
            plt = self.plt

        if pre_pos:
            pos = pre_pos
        else:
            pos = nx.kamada_kawai_layout(self.graph, scale=1)

        edge_width = []
        for edge in self.graph.edges(data=True):
            edge_width.append(edge[2]['weight'] * 0.1)

        stimulus = nx.get_node_attributes(self.graph, 's')

        if consider_stimulus:
            labels = {n: n if stimulus[n] >
                      self.render_label_size or force_text_rendering else '' for n in self.graph}
        else:
            labels = {n: n for n in self.graph}

        node_size = [stimulus[n] *
                     300 if consider_stimulus else 10 for n in self.graph]

        ax = plt.gca()
        # ax.set_xlim([-1.4, 1.4])
        # ax.set_ylim([-1.4, 1.4])
        ax.set_title(title)

        nx.draw_networkx_nodes(
            self.graph, pos=pos, node_size=node_size, node_color='none', edgecolors='red', linewidths=0.3)
        nx.draw_networkx_labels(self.graph, pos, font_size=6, labels=labels,
                                verticalalignment='bottom')
        nx.draw_networkx_edges(
            self.graph, pos, width=edge_width, arrowsize=arrow_size)
        plt.tight_layout()
        plt.savefig(path)
        plt.clf()

        return pos
