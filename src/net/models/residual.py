import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(torch.nn.Module):
    def __init__(self, prev_features_count, mid_count, next_features_count):
        super(Residual, self).__init__()

        self.lin1 = nn.Linear(prev_features_count, mid_count)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(mid_count, next_features_count)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(next_features_count, next_features_count)

    def forward(self, data):
        x, residual_input = data
        residual = x
        x = self.lin1(x)
        # x = F.dropout(x, p=0.5)
        x = self.relu1(x)
        x = self.lin2(x)
        # x = F.dropout(x, p=0.5)
        x = self.relu2(x)
        x = self.lin3(x)

        if residual_input != None:
            # x = self.relu2(x)
            x += residual_input
            # x = F.relu(x)

        # x = self.relu2(x)

        return x, residual


class ResidualModel(torch.nn.Module):
    def __init__(self, num_classes, steps=4, output_channels=1):
        super(ResidualModel, self).__init__()
        self.steps = steps

        self.downs = []
        self.ups = []
        self.flatten = nn.Flatten()
        current = num_classes * output_channels
        
        for i in range(steps):
            self.downs.append(Residual(current, int(
                current * 0.9), int(current * 0.8)))
            self.ups.insert(0, Residual(int(current * 0.8),
                            int(current * 0.9), current))
            current = int(current * 0.8)

        self.layers = torch.nn.ModuleList(self.downs+self.ups)
        self.last = torch.nn.Linear(int(num_classes * output_channels), output_channels)

    def forward(self, data):
        x = data
        downs = []
        x = self.flatten(x)
        for i in range(self.steps):
            x, residual = self.downs[i]((x, None))
            downs.insert(0, residual)

        for i in range(self.steps):
            x, _ = self.ups[i]((x, downs[i]))

        x = self.last(x)
        return x
