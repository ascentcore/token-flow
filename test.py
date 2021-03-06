from curses.panel import top_panel
import torch
from net.model import Model
from net.context import Context
from settings import path, edgelist, tokens, generate_length, preserve_history

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


context = Context(path)
graph = context.initialize_from_edgelist(
    path + f'/dataset/{edgelist}.txt-edgelist.txt')
context.G = graph
print(f'loading {path}/temp_model')
model = torch.load(f'{path}/dataset/temp_model')
model.eval()




text = ' '.join(tokens)


history = []

for token in tokens:
    context.stimulate_token(graph, token)
    token_index = context.get_token_index(token)
    history.append(token_index)

print("===========================================")
for i in range(generate_length):
    data = context.get_tensor_from_nodes(graph)
    data = data.to(device)
    output = model(data)
    # print(output)
    # predict_index = output.argmax()
    # print(output.shape)
    # print(predict_index)
    # predict_index = output.sum(dim=1).argmax()
    
    # _, pred = output.max(dim = 1)
    # print(_, pred)
    # output = output.squeeze(1)
    
    top_keys = torch.topk(output, k=preserve_history + 1).indices.tolist()
    top_keys = [x for x in top_keys if x not in history]
    #  print([context.vocabulary[key] for key in top_keys])
    predict_index = top_keys[0]

    token = context.translate(predict_index)
    text = text + ' ' + token
    
    context.decrease_stimulus(graph)
    lemma = context.get_lemma(token)
    
    context.stimulate_token(graph, lemma, debug=False)
    if lemma != token:
        context.stimulate_token(graph, token, debug=False)

    # # context.render(f'./output/sample.jpg', title=text)

    # if token == '<end>':
    #     context.decrease_stimulus(graph, 0.2)

    history.append(predict_index)
    history = history[-preserve_history:]

   

print(text)
