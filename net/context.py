import networkx as nx
import torch
from torch_geometric.data import Data


class Context:

    # Initial weight of the edge when created
    initial_weight = 0.1

    # Edge weight increase when connection exists
    weight_increase = 0.05

    # Degradation of signal from one node to the childre
    neuron_resistence = 0.75

    # Initial stimulus
    stimulus = 1

    # Stimulus propagation threshold
    propagate_threshold = 0.1

    # Stimulus decrease
    temp_decrease = 0.1

    def __init__(self, dictionary_path):
        self.keys = []
        G = nx.DiGraph()
        print(f'Loading context from {dictionary_path}')
        with open(f'{dictionary_path}/dataset/dictionary.txt', 'r') as fp:
            index = 0
            for line in fp:
                x = line[:-1]
                self.keys.append(x)
                G.add_node(index, s=0)
                index = index+1

        print(f'Context initialized with {len(self.keys)} nodes')
        self.G = G

    def get_tensor_from_nodes(self, data_graph, next_token=None):
        nodes = data_graph.nodes(data=True)
        edges = data_graph.edges(data=True)

        n_nodes = len(self.keys)

        y0 = [edge[0] for edge in edges]
        y1 = [edge[1] for edge in edges]
        weights = [edge[2]['weight'] for edge in edges]

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
        return self.keys.index(token)

    def create_links(self, graph, token_list):
        previous = None
        for token in token_list:
            token_index = self.get_token_index(token)
            if previous is not None:
                if graph.has_edge(previous, token_index):
                    graph[previous][token_index]['weight'] += self.weight_increase
                else:
                    graph.add_edge(previous, token_index,
                                   weight=self.initial_weight)
            previous = token_index

    def stimulate_token(self, graph, token, stimulus=stimulus):
        token_index = self.get_token_index(token)
        to_set = self.stimulate(graph, token_index, stimulus, {})
        # print([(self.keys[idx], str(round(to_set[idx]['s'], 3)))
        #       for idx in to_set.keys()])
        nx.set_node_attributes(graph, to_set)

    def decrease_stimulus(self, graph, decrease=temp_decrease):
        for node in graph.nodes():
            graph.nodes[node]['s'] = max(0, graph.nodes[node]['s'] - decrease)

    def stimulate(self, graph, token_index, stimulus=stimulus, to_set={}):
        if token_index not in to_set.keys():
            node = graph[token_index]
            current_stimulus = graph.nodes[token_index]['s'] + stimulus
            to_set[token_index] = {'s': min(1, current_stimulus)}

            if stimulus > self.propagate_threshold:
                for sub_key in node.keys():

                    if sub_key != token_index:
                        weight = node[sub_key]['weight']
                        self.stimulate(graph, sub_key,
                                       weight * stimulus * self.neuron_resistence, to_set=to_set)

        return to_set
