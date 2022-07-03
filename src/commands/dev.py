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


def stimulate(key, value, G, visited=[]):
    value = float(value)
    if key not in visited:
        visited.append(key)
        node = G[key]
        previous_value = G.nodes(data=True)[key]['s']
        new_value = min(1, previous_value + value)
        nx.set_node_attributes(G, {key: {'s':  new_value}})

        if (new_value < 0.01):
            return
        
        # for sub_key in node.keys():
        #     if sub_key != key and sub_key not in visited:
        #         weight = node[sub_key]['weight']
        #         stimulate(sub_key, weight * new_value, G, visited)

    return visited


def do_plot(G, writer, pos = None):
    threshold = 0.05

    

    nodes = (
        node
        for node, data
        in G.nodes(data=True)
        if data.get("s") > threshold
    )

    subgraph = G.subgraph(nodes)

    stimulus = nx.get_node_attributes(subgraph, 's')

    node_alpha = []

    for n, v in stimulus.items():
        node_alpha.append(v)

    edge_width = []
    for edge in subgraph.edges(data=True):
        edge_width.append(edge[2]['weight'])


    if pos is None:
        pos = nx.spring_layout(subgraph, k=0.15, iterations=100)

    nx.draw_networkx_nodes(
        subgraph, pos=pos, node_size=10, alpha=node_alpha)
    nx.draw_networkx_labels(subgraph, pos, font_size=6,
                            verticalalignment='bottom')
    nx.draw_networkx_edges(subgraph, pos, width=edge_width)

    plt.savefig('output/graph.jpg')
    plt.clf()
    image = imageio.imread('output/graph.jpg')
    writer.append_data(image)

    nx.write_gml(subgraph, 'output/graph.gml')


def execute(args):
    G = nx.read_edgelist("graphs/full-dictionary.edgelist")
    G.remove_edges_from(nx.selfloop_edges(G))
    # pos = nx.spring_layout(G, k=0.15, iterations=100)
    # pos = nx.kamada_kawai_layout(G)
    pos = None

    nx.set_node_attributes(G, 0, name='s')

    test_text = """
    The lion (Panthera leo) is a large cat of the genus Panthera native to Africa and India. It has a muscular, broad-chested body, short, rounded head, round ears, and a hairy tuft at the end of its tail. It is sexually dimorphic; adult male lions are larger than females and have a prominent mane. It is a social species, forming groups called prides. A lion's pride consists of a few adult males, related females, and cubs. Groups of female lions usually hunt together, preying mostly on large ungulates. The lion is an apex and keystone predator; although some lions scavenge when opportunities occur and have been known to hunt humans, the species typically does not actively seek out and prey on humans.
    The lion inhabits grasslands, savannas and shrublands. It is usually more diurnal than other wild cats, but when persecuted, it adapts to being active at night and at twilight. During the Neolithic period, the lion ranged throughout Africa, Southeast Europe, the Caucasus, Western Asia and northern parts of India, but it has been reduced to fragmented populations in sub-Saharan Africa and one population in western India. It has been listed as Vulnerable on the IUCN Red List since 1996 because populations in African countries have declined by about 43% since the early 1990s. Lion populations are untenable outside designated protected areas. Although the cause of the decline is not fully understood, habitat loss and conflicts with humans are the greatest causes for concern.
    One of the most widely recognised animal symbols in human culture, the lion has been extensively depicted in sculptures and paintings, on national flags, and in contemporary films and literature. Lions have been kept in menageries since the time of the Roman Empire and have been a key species sought for exhibition in zoological gardens across the world since the late 18th century. Cultural depictions of lions were prominent in Ancient Egypt, and depictions have occurred in virtually all ancient and medieval cultures in the lion's historic and current range.
    """

    doc = nlp(test_text.lower())
    sentences = str(doc).splitlines()
    with imageio.get_writer('output/output.gif', mode='I') as writer:
        for sentence in sentences:
            if sentence != '':
                tokens = nlp(sentence)
                for token in tokens:
                    if token.pos_ == "NOUN":
                        # or token.pos_ == "PROPN" or token.pos_ == "VERB":
                        try:
                            stimulate(token.lemma_, 0.1, G)
                        except KeyError:
                            logger.info(f'Unknown token {token.lemma_}')

        

            # list = G.nodes(data=True)
            # set_values = {}
            # for node, data in list:
            #     set_values[node] = {'s': data['s'] - 0.02}
            # nx.set_node_attributes(G, set_values)

                do_plot(G, writer, pos)
