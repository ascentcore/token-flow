import torch
import networkx as nx
from torch_geometric.nn import TransformerConv, GCNConv, Linear
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        nhid = 256
        """ GCNConv layers """
        self.conv1 = GCNConv(num_features, nhid)
        # self.conv2 = GCNConv(nhid, nhid)
        self.conv3 = GCNConv(nhid, num_features)

        # self.lin1 = Linear(nhid, nhid)
        # self.lin2 = Linear(nhid, nhid//2)
        # self.lin3 = Linear(nhid//2, num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        
        # x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        # x = F.relu(self.lin3(x))

        return torch.sigmoid(x)

