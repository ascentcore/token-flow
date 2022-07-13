from sklearn.metrics import roc_auc_score
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, SAGEConv, SGConv
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_geometric.loader import DataLoader
import numpy as np

from net.dataset import ContextualGraphDataset
from net.model import GCN
from settings import path, clear_dataset, train_epochs

dataset = ContextualGraphDataset(
    source=path, from_scratch=clear_dataset)
dataset = dataset.shuffle()
one_tenth_length = int(len(dataset) * 0.1)
train_dataset = dataset[:one_tenth_length * 8]
val_dataset = dataset[one_tenth_length*8:one_tenth_length * 9]
test_dataset = dataset[one_tenth_length*9:]


batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size)


# number of graphs
print("Number of graphs: ", len(dataset))

# number of features
print("Number of features: ", dataset.num_features)

# number of classes
print("Number of classes: ", dataset.num_classes)

print("X shape: ", dataset[0].x.shape)
print("Edge shape: ", dataset[0].edge_index.shape)
print("Y shape: ", dataset[0].y.shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(dataset.num_features).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
crit = torch.nn.CrossEntropyLoss()
# crit = torch.nn.MSELoss()

def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze(1)
        label = data.y.to(device)
        loss = crit(output, label.to(torch.float32))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        
    return loss_all / len(train_dataset)


for epoch in range(1, train_epochs):
    loss = train()
    print(f'Epoch: {epoch}, Loss: {loss}')
    torch.save(model, f'{path}/dataset/temp_model')

torch.save(model, f'{path}/dataset/trained_model')
