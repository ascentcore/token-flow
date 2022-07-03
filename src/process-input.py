import csv
import spacy
import networkx as nx
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

with open('defs.txt') as f:
    text = f.read()

doc = nlp(text)
sentences = str(doc).splitlines()

nouns_list = []
for sentence in sentences:
    if sentence != '':
        nouns = []
        tokens = nlp(str(sentence).lower())
        for token in tokens:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                nouns.append(token.lemma_)
        nouns = list(dict.fromkeys(nouns))
        nouns_list.append(nouns)

dicts_list = []
for lst in nouns_list:
    dict = {}
    for noun in lst:
        dict[noun] = []
        for n in lst:
            if noun != n:
                dict[noun].append(n)
    dicts_list.append(dict)

def merge_two_dicts(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
            dict_3[key] = value + dict_1[key]
   return dict_3
   
dict_graph = {}
for dict in dicts_list:
    dict_graph = merge_two_dicts(dict_graph, dict)

g = nx.Graph()
for k, vs in dict_graph.items():
    for v in vs:
        g.add_edge(k,v)

plt.figure(1, figsize=(50, 30), dpi=60)
pos = nx.spring_layout(g, k=0.20, iterations=20)
nx.draw(g, pos, with_labels = True, node_color='pink', node_size=7000, font_size=18)
plt.savefig('graph.jpg')

# Extract edges and add weights

def generate_edges(graph): 
    graph_edges = []
    for node in graph: 
        for adjacent_node in graph[node]:
            graph_edges.append([node, adjacent_node])
    return graph_edges

def assign_weights(edges):
    weighted_graph = []
    for edge in edges:
        weighted_graph.append([edge[0], edge[1], 0.1])
    return weighted_graph

graph_edges = generate_edges(dict_graph)
weighted_graph = assign_weights(graph_edges)
weighted_graph = [[i, weighted_graph.count(i)] for i in weighted_graph]

res = []
[res.append(x) for x in weighted_graph if x not in res]

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

weighted_graph_fin = []
for r in res:
    weighted_edge = r[0]
    cnt = r[1] - 1
    increment = 0.05 * cnt
    weighted_edge[2] += increment
    weighted_edge[2] = truncate(weighted_edge[2], 2)
    weighted_graph_fin.append(weighted_edge)
    
fields = ['Source', 'Target', 'Label']
with open('edges.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(weighted_graph_fin)
