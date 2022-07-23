import torch
import torch_geometric.nn as tgnn
import torch.nn.functional as F
import torch_geometric as tg
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Model, self).__init__()
        torch.manual_seed(12345)
        channels = 1
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=0, end_dim=-1))

        n_channels = self.feature_extractor(torch.empty(
            1, channels, num_classes, num_classes)).size(-1)

        self.net_classifier = nn.Sequential(
            nn.Linear(n_channels, 200),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100, num_classes))

        self.token_classifier = nn.Sequential(
            nn.Linear(num_classes * 2, 200),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100, num_classes)
        )


    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = tg.utils.to_dense_adj(edge_index)
        # print(x.shape)

        features = self.feature_extractor(x)
        # print('Flatten output of feature extractor: ', features.shape)
        out = self.net_classifier(features)
        
        out = torch.cat((out, data.x.squeeze(1)), dim=0).unsqueeze(dim = 0)
        

        token_out = self.token_classifier(out)
        token_out = token_out.squeeze(dim = 0)

        # print('Classifier output: ', out.shape)
        # out = F.log_softmax(out, dim=-1)

        return token_out