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
        # x = F.relu(x)
         
       
        
        if residual_input != None:
            # x = self.relu2(x)       
            x += residual_input
            # x = F.relu(x)

        return x, residual


class ResidualModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResidualModel, self).__init__()
        
        self.res1 = Residual(num_classes, int(num_classes*0.75), int(num_classes / 2))
        self.res2 = Residual(int(num_classes/2), int(num_classes*0.5), int(num_classes / 4))
        self.res3 = Residual(int(num_classes/4), int(num_classes*5), int(num_classes / 2))
        self.res4 = Residual(int(num_classes/2), int(num_classes*0.75), num_classes)

    def forward(self, data):
        x = data
        x, residual_1 = self.res1((x, None))
        x, residual_2 = self.res2((x, None))
        x, residual_3 = self.res3((x, residual_2))
        x, residual_4 = self.res4((x, residual_1))
        return x
