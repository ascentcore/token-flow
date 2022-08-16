import torch
import torch.nn.functional as F

class AE(torch.nn.Module):
    def __init__(self, vocab_size, output_channels=1):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(int(vocab_size), int(vocab_size / 2)),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(int(vocab_size / 2), int(vocab_size / 4)),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(int(vocab_size / 4), int(vocab_size / 6))
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(int(vocab_size / 6), int(vocab_size / 4)),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(int(vocab_size / 4), int(vocab_size / 2)),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(int(vocab_size / 2), int(vocab_size)),
            torch.nn.Sigmoid()
        )


        self.lin1 = torch.nn.Linear(vocab_size, vocab_size)
        self.lin2 = torch.nn.Linear(vocab_size, vocab_size)
        self.lin3 = torch.nn.Linear(vocab_size, vocab_size)
        self.lin4 = torch.nn.Linear(vocab_size, vocab_size)

    def forward(self, x):
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        x = F.sigmoid(x)
        return x
