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

    def forward(self, data):
        x, residual_input = data
        residual = x
        x = self.lin1(x)
        # x = F.dropout(x, p=0.5)
        x = self.relu1(x)
        x = self.lin2(x)
        # x = F.dropout(x, p=0.5)
        # x = self.relu2(x)

        if residual_input != None:
            # x = self.relu2(x)
            x += residual_input
            # x = F.relu(x)

        x = self.relu2(x)

        return x, residual


class ResidualModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResidualModel, self).__init__()

        self.res1_down = Residual(num_classes, int(
            num_classes*0.9), int(num_classes * 0.8))
        self.res2_down = Residual(int(num_classes * 0.8),
                                  int(num_classes*0.7), int(num_classes * 0.6))
        self.res3_down = Residual(int(num_classes * 0.6),
                                  int(num_classes*0.5), int(num_classes * 0.4))

        self.res3_up = Residual(int(num_classes * 0.4),
                                int(num_classes*0.5), int(num_classes * 0.6))
        self.res2_up = Residual(int(num_classes * 0.6),
                                int(num_classes*0.7), int(num_classes * 0.8))
        self.res1_up = Residual(int(num_classes * 0.8),
                                int(num_classes*0.9), num_classes)

        self.last = torch.nn.Sigmoid()

    def forward(self, data):
        x = data
        x, residual_1 = self.res1_down((x, None))
        x, residual_2 = self.res2_down((x, None))
        x, residual_3 = self.res3_down((x, None))
        x, _ = self.res3_up((x, residual_3))
        x, _ = self.res2_up((x, residual_2))
        x, _ = self.res1_up((x, residual_1))
        # x = self.last(x)
        return x
