from net.context import Context
import torch
from torch.functional import F
import math
import torch_geometric.nn as tgnn
from torch.nn import Dropout, Linear, Sequential
from torch.utils.data import DataLoader

tgnn.GATv2Conv


class Model(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Model, self).__init__()
        torch.manual_seed(12345)
        hidden_dim = 8
        hidden_dim = 32
        num_heads = 2
        out_heads = 1
        self.conv1 = tgnn.GATv2Conv(
            num_features, hidden_dim, num_heads)
        self.conv2 = tgnn.GATv2Conv(
            hidden_dim * num_heads, num_classes, heads = out_heads, concat = False)

        self.lin1 = tgnn.Linear(hidden_dim * num_heads, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        # x = tgnn.global_mean_pool(x,batch)
        # x = self.lin1(x)
        # print(x)
        x = torch.mean(x, dim=1)
        x = F.log_softmax(x)
        # print(x)
        return x


def run_sample():
    print('Running sample 1')

    text = """
    A car is a vehicle. The vehicle runs on roads.
    """

    context = Context()
    context.from_text(
        text,
        connect_all=True,
        set_graph=True)

    # print(context.vocabulary)

    dataset = []
    print('')
    print('##############################################################')
    print('#                           Dataset                          #')
    print('##############################################################')
    print('')
    print(context.vocabulary)
    print('')

    for token in ['<start>', 'a', 'car', 'is', 'a', 'vehicle', '<end>', 'the', 'vehicle', 'runs', 'on', 'roads', '<end>']:
        data = context.get_tensor_from_nodes(context.G, (token, token), with_state=True)
        # context.decrease_stimulus(context.G)
        # context.stimulate_token(context.G, token, debug=True)
        dataset.append(data)
        print(data.x.squeeze(1))
        print(data.y)
        print('-------------------------------------------------------')

    print('')
    print('##############################################################')
    print('#                            Model                           #')
    print('##############################################################')
    print('')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(num_features=1, num_classes=len(context.vocabulary))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()

    model.train()
    for i in range(0, 50):
        model.train()
        loss_all = 0
        for data in dataset:
            data = data.to(device)

            output = model(data)
            label = data.y

            loss = crit(output, label.to(torch.float32))
            loss_all += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch: {i} Loss: {loss_all:.3f}')

    context.decrease_stimulus(context.G, 1)
    token = 'car'
    # for token in ['<start>', 'a', 'car']:
    for i in range(0, 10):
        context.decrease_stimulus(context.G)
        context.stimulate_token(context.G, token)
        data = context.get_tensor_from_nodes(context.G)

        output = model(data)
        predict_index = output.argmax()
        print(output, predict_index)
        print(token, '->', context.vocabulary[predict_index])
        token = context.vocabulary[predict_index]
