import torch
import networkx as nx
from torch_geometric.nn import TransformerConv, GCNConv, DenseGCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        """ GCNConv layers """
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # return F.log_softmax(x, dim=1)
        return torch.sigmoid(x)
