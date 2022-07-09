import torch
import networkx as nx
from torch_geometric.data import Data
from net.model import GCN
from net.dataset import stimulate, get_initial_graph, get_dictionary_keys, stimulate_graph_with_text, get_tensor_from_nodes

import spacy
nlp = spacy.load("en_core_web_sm")

model = torch.load('./tempmodel')
model.eval()

keys = get_dictionary_keys()
G = nx.read_edgelist('datasets/kids/dataset/graphs/environment.txt')

# graph = nx.relabel_nodes(G, mapping)


graph = get_initial_graph(keys)
for edge in G.edges(data=True):
    graph.add_edge(int(edge[0]), int(edge[1]), weight=edge[2]['weight'])


nx.set_node_attributes(graph, 0, 's')
text = "<start> The wetlands biome is a combination of land and water"


stimulate_graph_with_text(text, keys, graph)

last = None
past_5 = []
for i in range(0, 20):
    data = get_tensor_from_nodes(graph, keys)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    output = model(data)
    output = output.squeeze(1)

    top_keys = torch.topk(output, k=10).indices.tolist()

    top_keys = [x for x in top_keys if x not in past_5]

    last = top_keys[0]
    past_5.append(last)
    past_5 = past_5[-5:]

    print(
        f'{text} -> {[keys[index] for index in top_keys]}')
    text = text + ' ' + keys[last]

    to_set = stimulate(int(last), 1, graph)
    nx.set_node_attributes(graph, to_set)
