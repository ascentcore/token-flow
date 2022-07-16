import torch
import torch_geometric.nn as tgnn
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Model, self).__init__()
        torch.manual_seed(12345)
        hidden_dim = 8
        hidden_dim = 32
        num_heads = 6
        out_heads = 1
        self.conv1 = tgnn.GATv2Conv(
            num_features, hidden_dim, num_heads)
        self.conv2 = tgnn.GATv2Conv(
            hidden_dim * num_heads, num_classes, heads = out_heads, concat = False)

        self.lin1 = tgnn.Linear(hidden_dim * num_heads, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        # x = tgnn.global_mean_pool(x,batch)
        # x = self.lin1(x)
        # print(x)
        x = torch.mean(x, dim=1)
        x = F.log_softmax(x)
        # print(x)
        return x
