import sys
import os
import logging
import spacy
import networkx as nx


logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")


def merge_two_dicts(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value + dict_1[key]
    return dict_3


def parse_file(path, G):
    with open(path) as f:
        text = f.read()

        
        key = None
        definition = None
        for line in text.splitlines():
            if key is None:
                key = line
                print(key)
            elif definition is None:
                definition = line
            else:
                doc = nlp(definition)
                sentences = str(doc).splitlines()

                tokens = {}
                G.add_node(key)
                for sentence in sentences:
                    if sentence != '':
                        tokens = nlp(str(sentence).lower())                        
                        for token in tokens:
                            if token.pos_ == "NOUN": 
                                #or token.pos_ == "PROPN":
                                # or token.pos_ == "VERB":
                                edge = G.get_edge_data(key, token.lemma_)                            
                                if edge is None:
                                    edge = G.add_edge(
                                        key, token.lemma_, weight=0.1)
                                else:
                                    edge.update({'weight': edge['weight'] + 0.01})

                key = None
                definition = None


def execute(args):
    if len(args) != 1:
        print("Usage: digest <path>")
        sys.exit(1)

    path = args[0]

    G = nx.Graph()

    logger.info(f'Digesting {path}')
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            parse_file(os.path.join(root, file), G)


    nx.write_edgelist(G, "output.edgelist")
