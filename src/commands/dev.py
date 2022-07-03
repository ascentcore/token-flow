import logging
from operator import sub
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import spacy
import imageio

nlp = spacy.load("en_core_web_sm")
figure(figsize=(8, 6), dpi=300)
logger = logging.getLogger(__name__)

# if key not in visited:
#     visited.append(key)
#     node = G[key]
#     if nodes is None:
#         nodes = G.nodes(data=True)
#     previous_value = nodes[key]['s']
#     new_value = min(1, previous_value + value)
#     # nx.set_node_attributes(G, {key: {'s':  new_value}})

#     if (new_value < 0.01):
#         return

#     for sub_key in node.keys():
#         if sub_key != key and sub_key not in visited:
#             weight = node[sub_key]['weight']
#             stimulate(sub_key, weight * new_value, G, visited, nodes=nodes)

# return visited


def do_plot(G, writer, pos=None, title=None):
    threshold = 0.1

    nodes = (
        node
        for node, data
        in G.nodes(data=True)
        if data.get("s") > threshold
    )

    subgraph = G.subgraph(nodes)

    stimulus = nx.get_node_attributes(subgraph, 's')

    node_alpha = []
    node_size = []

    for n, v in stimulus.items():
        node_alpha.append(v)
        node_size.append(700 * v)

    edge_width = []
    for edge in subgraph.edges(data=True):
        edge_width.append(edge[2]['weight'])

    if pos is None:
        # pos = nx.spring_layout(subgraph, k=0.15, iterations=100)
        pos = nx.kamada_kawai_layout(subgraph)

    nx.draw_networkx_nodes(
        subgraph, pos=pos, node_size=node_size, node_color='none', edgecolors='red', linewidths=0.1)
    nx.draw_networkx_labels(subgraph, pos, font_size=6,
                            verticalalignment='bottom')
    nx.draw_networkx_edges(subgraph, pos, width=edge_width)
    ax = plt.gca()
    ax.set_title(title)
    plt.savefig('output/graph.jpg')
    plt.clf()
    image = imageio.imread('output/graph.jpg')
    writer.append_data(image)

    nx.write_gml(subgraph, 'output/graph.gml')


def stimulate(key, value, G, factor=1, to_set=None, nodes=None):
    value = float(value) * factor

    if to_set is None:
        to_set = {}
    if nodes is None:
        nodes = G.nodes()

    if key not in to_set:
        node = G[key]
        previous_value = nodes[key]['s']
        new_value = min(1, previous_value + value)
        to_set[key] = {'s':  new_value}

        if new_value > 0.1:
            for sub_key in node.keys():
                if sub_key != key:
                    weight = node[sub_key]['weight']
                    stimulate(sub_key, weight * new_value,
                              G, factor * 0.5, to_set=to_set, nodes=nodes)

    return to_set


def execute(args):
    G = nx.read_edgelist("graphs/full-dictionary.edgelist")
    G.remove_edges_from(nx.selfloop_edges(G))
    # pos = nx.spring_layout(G, k=0.15, iterations=100)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.random_layout(G)
    pos = None

    nx.set_node_attributes(G, 0, name='s')

    print(f'Reading from {args[0]}')
    with open(args[0]) as f:
        test_text = f.read()

    doc = nlp(test_text.lower())
    sentences = str(doc).splitlines()
    with imageio.get_writer('output/output.gif', mode='I') as writer:
        for sentence in sentences:
            if sentence != '':
                to_link = []
                rest_tokens = []
                tokens = nlp(sentence)
                for token in tokens:
                    if token.pos_ == "NOUN":
                        if not G.has_node(token.lemma_):
                            G.add_node(token.lemma_, s=0)
                            to_link.append(token.lemma_)

                        if token.lemma_ not in rest_tokens:
                            rest_tokens.append(token.lemma_)

                        to_set = stimulate(token.lemma_, 0.5, G)
                        nx.set_node_attributes(G, to_set)

                    elif token.pos == "PROPN":
                        print(token.pos_)

                for token in to_link:
                    for to_token in rest_tokens:
                        if token != to_token:
                            G.add_edge(token, to_token, weight=0.1)

                list = G.nodes(data=True)
                set_values = {}
                for node, data in list:
                    new_value = data['s'] - 0.05
                    set_values[node] = {'s': max(0, new_value)}
                nx.set_node_attributes(G, set_values)

                #We should create connections between nodes that are strongly stimulated
                nodes = (
                    node
                    for node, data
                    in G.nodes(data=True)
                    if data.get("s") > 0.5
                )

                subgraph = G.subgraph(nodes)
                nodes_list = subgraph.nodes()

                if len(nodes_list) > 1:
                    
                    for node_i in nodes_list:
                        for node_j in nodes_list:
                            if node_i != node_j and not G.has_edge(node_i, node_j):
                                print(node_i, node_j)
                                G.add_edge(node_i, node_j, weight=0.1)
                # End of we should create :)
                
                

                do_plot(G, writer, pos, title=sentence)
