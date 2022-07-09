import torch
import networkx as nx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        """ GCNConv layers """
        self.conv1 = GCNConv(num_features, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, num_features)

    def forward(self, data):
        x, edge_index, y = data.x, data.edge_index, data.y

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))

        return torch.sigmoid(x)
        # return F.log_softmax(x, dim=1)


