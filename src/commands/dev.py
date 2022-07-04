import logging
from operator import sub
from random import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import spacy
import imageio

nlp = spacy.load("en_core_web_sm")
plt = figure(figsize=(8, 6), dpi=150)

logger = logging.getLogger(__name__)

relink = True
factor_constant = 0.75
render_filter_value = 0.1
stimulus = 0.5
temp_decrease = 0.05
edge_decrease = 0.005

positions = {}


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
        edge_width.append(edge[2]['weight'] * 0.5)

    if pos is None:
        pos = nx.fruchterman_reingold_layout(subgraph)
        # pos = nx.spring_layout(subgraph, k=0.5,   iterations=100)
        for p_key in pos.keys():
            if p_key in positions:
                pos[p_key] = positions[p_key]
            else:
                positions[p_key] = pos[p_key]

    nx.draw_networkx_nodes(
        subgraph, pos=pos, node_size=node_size, node_color='none', edgecolors='red', linewidths=0.1)
    nx.draw_networkx_labels(subgraph, pos, font_size=6,
                            verticalalignment='bottom')
    nx.draw_networkx_edges(subgraph, pos, width=edge_width)
    ax = plt.gca()
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_title(title)
    plt.savefig('output/graph.jpg')
    plt.clf()
    image = imageio.imread('output/graph.jpg')
    writer.append_data(image)

    # nx.write_gml(subgraph, 'output/graph.gml')


def stimulate(key, value, G, factor=1, to_set=None, nodes=None):
    value = float(value) * factor

    if to_set is None:
        to_set = {}
    if nodes is None:
        nodes = G.nodes()

    if key not in to_set:
        node = G[key]
        previous_value = nodes[key]['s']
        new_value = previous_value + value
        to_set[key] = {'s':  new_value}

        if new_value > 0.1:
            for sub_key in node.keys():
                if sub_key != key:
                    weight = node[sub_key]['weight']
                    stimulate(sub_key, weight * new_value,
                              G, factor * factor_constant, to_set=to_set, nodes=nodes)

    return to_set


def execute(args):
    G = nx.read_edgelist("graphs/full-dictionary.edgelist")
    G.remove_edges_from(nx.selfloop_edges(G))
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
                to_set = {}
                for token in tokens:
                    if token.pos_ == "NOUN":
                        if not G.has_node(token.lemma_):
                            node = G.add_node(token.lemma_, s=0)
                            to_link.append(token.lemma_)

                        if token.lemma_ not in rest_tokens:
                            rest_tokens.append(token.lemma_)

                        to_set = stimulate(token.lemma_, stimulus, G, to_set=to_set)
                        nx.set_node_attributes(G, to_set)

                if relink:
                    for token in to_link:
                        for to_token in rest_tokens:
                            if token != to_token:
                                G.add_edge(token, to_token, weight=0.1)

               

                # We should create connections between nodes that are strongly stimulated

                if relink:
                    nodes = (
                        node
                        for node, data
                        in G.nodes(data=True)
                        if data.get("s") > render_filter_value
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



                ### Cleanup ####
                # temp_decrease

                list = G.nodes(data=True)
                set_values = {}
                for node, data in list:
                    new_value = data['s'] - temp_decrease
                    set_values[node] = {'s': max(0, new_value)}
                nx.set_node_attributes(G, set_values)


                # edge_decrease

                # edges = G.edges(data=True)
                # set_values = {}
                # for edge, data in edges:
                #     new_value = data['weight'] - edge_decrease
                #     set_values[edge] = {'weight': max(0, new_value)}
                # nx.set_edge_attributes(G, set_values)
