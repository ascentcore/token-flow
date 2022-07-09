from curses.panel import top_panel
import torch
from net.model import GCN
from net.context import Context
from settings import path, edgelist, tokens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


context = Context(path)

graph = context.initialize_from_edgelist(
    path + f'/dataset/{edgelist}.txt-graph.edgelist')
print(f'loading {path}/temp_model')
model = torch.load(f'{path}/temp_model')
model.eval()

text = ' '.join(tokens)

preserve_history = 20
history = []

for token in tokens:
    context.stimulate_token(graph, token)
    token_index = context.get_token_index(token)
    history.append(token_index)


for i in range(100):

    data = context.get_tensor_from_nodes(graph)
    data = data.to(device)
    output = model(data)
    output = output.squeeze(1)
    top_keys = torch.topk(output, k=preserve_history + 1).indices.tolist()
    top_keys = [x for x in top_keys if x not in history]

    last = top_keys[0]
    token = context.translate(last)
    text = text + ' ' + token

    context.stimulate_token(graph, token, debug=False)
    history.append(last)
    history = history[-preserve_history:]

    if token == '<end>':
        context.decrease_stimulus(graph, 0.5)

print(text)
