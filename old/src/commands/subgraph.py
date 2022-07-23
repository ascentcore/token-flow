import sys
import json
import os
import logging
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt = figure(figsize=(10, 10), dpi=150)


def execute(args):

    if len(args) != 2:
        print("Usage: subgraph <path> <output>")
        sys.exit(1)

    path = args[0]
    output = args[1]

    logger = logging.getLogger(__name__)
    nlp = spacy.load("en_core_web_sm")

    with open('./text-definitions/dictionary/output.json') as f:
        data = f.read()

    dictionary = json.loads(data)

    G = nx.DiGraph()

    def parse_and_append(text, initial=False):
        doc = nlp(text)

        for sentence in doc.sents:
            previous = None
            for token in sentence:
                lemma = token.lemma_

                if token.lemma_ in dictionary:

                    if lemma not in G.nodes():
                        G.add_node(lemma)

                        if previous is not None:
                            G.add_edge(previous, lemma, weight=0.1)

                            if G.has_edge(previous, lemma):
                                G[previous][lemma]['weight'] += 0.01
                            else:
                                G.add_edge(previous, lemma, weight=0.1)

                        if initial:
                            parse_and_append(dictionary[lemma], initial=False)

                    previous = lemma
                elif token.pos_ == "PROPN":
                    G.add_node(lemma)
                    previous = lemma

                else:
                    print(token.lemma_)

    with open(path) as f:
        text = f.read()
        parse_and_append(text, initial=True)

    # pos = nx.spring_layout(G)
    pos = nx.fruchterman_reingold_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=1)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True, width=[
                           G[u][v]['weight'] for u, v in G.edges()])
    # plt.show()
    plt.savefig(f'output/{output}.jpg')
    nx.write_gml(G, f'graphs/{output}.gml')
