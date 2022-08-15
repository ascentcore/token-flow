import torch


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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
