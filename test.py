from curses.panel import top_panel
import torch
from net.model import GCN
from net.context import Context

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = 'datasets/simple'
context = Context(path)

graph = context.initialize_from_edgelist(
    path + '/dataset/car.txt-graph.edgelist')

model = torch.load(f'{path}/temp_model')
model.eval()
text = ''
token = '<start>'
past_5 = []


for i in range(5):
    token_index = context.get_token_index(token)
    past_5.append(token_index)
    context.stimulate_token(graph, token)

    data = context.get_tensor_from_nodes(graph)
    data = data.to(device)
    output = model(data)
    output = output.squeeze(1)
    top_keys = torch.topk(output, k=10).indices.tolist()
    top_keys = [x for x in top_keys if x not in past_5]

    print(top_keys, past_5)
    
    last = top_keys[0]
    token = context.translate(last)
    text = text + ' ' + token
    print(text)

