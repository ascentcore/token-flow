import logging
from operator import sub
from random import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import spacy
import imageio

plt = figure(figsize=(6, 6), dpi=150)
logger = logging.getLogger(__name__)


def merge_two_dicts(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value + dict_1[key]
    return dict_3


def do_draw(G):

    nodes = (
        node
        for node, data
        in G.nodes(data=True)
        if data.get("s") > 0.2
    )

    subgraph = G.subgraph(nodes)

    render_label_size = 0.2
    stimulus = nx.get_node_attributes(subgraph, 's')

    node_alpha = []
    node_size = []

    for n, v in stimulus.items():
        node_alpha.append(v)
        node_size.append((700 if v > render_label_size else 100) * v)

    edge_width = []
    for edge in subgraph.edges(data=True):
        edge_width.append(edge[2]['weight'] *
                          (0.5 if v > render_label_size else 0.1))

    labels = {n: n if stimulus[n] >
              render_label_size else '' for n in subgraph}

    # pos = nx.spring_layout(subgraph, k=0.1, iterations=50)
    pos = nx.circular_layout(subgraph)
    nx.draw_networkx_nodes(
        subgraph,
        node_size=node_size,
        pos=pos,
        node_color='none',
        edgecolors='red',
        linewidths=0.1)

    nx.draw_networkx_labels(subgraph, pos, font_size=6,
                            labels=labels,
                            verticalalignment='bottom')
    nx.draw_networkx_edges(subgraph, pos, width=edge_width)
    # ax = plt.gca()
    # ax.set_xlim([-1.2, 1.2])
    # ax.set_ylim([-1.2, 1.2])
    # ax.set_title(title)
    plt.savefig('output/graph.jpg')
    plt.clf()


def parse_file(file, G, nlp):

    initial_stimulus = 0.4
    stimulus = 0.1
    temp_decrease = 0.05
    factor_constant = 0.75

    def do_link(from_token, to_token):
        from_tkn = from_token.lemma_.lower()
        to_tkn = to_token.lemma_.lower()
        if from_tkn not in G.nodes():
            G.add_node(from_tkn, s=initial_stimulus)
       
        if to_tkn not in G.nodes():
            G.add_node(to_tkn, s=initial_stimulus)
       
        if G.has_edge(from_tkn, to_tkn):
            G[from_tkn][to_tkn]['weight'] += 0.01
        else:
            G.add_edge(from_tkn, to_tkn, weight=0.01)


    
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

    with open(file) as f:
        text = f.read()

        doc = nlp(text)

        for sentence in doc.sents:
            print(' --------- New Sentence ---------')
            print(sentence)
            print(' --------------------------------')
            to_stimulate = []
            for chunk in sentence.noun_chunks:
                print(chunk.text, '|', chunk.root.text, '|', chunk.root.dep_, '|',
                      chunk.root.head.text)
                do_link(chunk.root, chunk.root.head)

                if chunk.root.lemma_ not in to_stimulate:
                    to_stimulate.append(chunk.root.lemma_.lower())
                
                if chunk.root.head.lemma_ not in to_stimulate:
                    to_stimulate.append(chunk.root.head.lemma_.lower())

            
            to_set = {}
            for token in to_stimulate:
                 to_set = stimulate(token, stimulus, G, to_set=to_set)
                 nx.set_node_attributes(G, to_set)


            do_draw(G)

            ### Cleanup ####
            # temp_decrease

            list = G.nodes(data=True)
            set_values = {}
            for node, data in list:
                new_value = data['s'] - temp_decrease
                set_values[node] = {'s': max(0, new_value)}
            nx.set_node_attributes(G, set_values)

        nx.write_edgelist(G, "output/sample.edgelist")


def execute(args):

    logger = logging.getLogger(__name__)
    nlp = spacy.load("en_core_web_sm")

    if len(args) != 1:
        print("Usage: digest <file>")
        sys.exit(1)

    file = args[0]

    G = nx.Graph()

    parse_file(file, G, nlp)
