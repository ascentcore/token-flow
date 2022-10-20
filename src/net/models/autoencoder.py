import torch
import torch.nn.functional as F


class AE(torch.nn.Module):
    def __init__(self, vocab_size, output_channels=1):
        super().__init__()
        self.output_channels = output_channels
        vocab_size = vocab_size * output_channels

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(int(vocab_size), int(vocab_size / 2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(int(vocab_size / 2), int(vocab_size / 4)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(int(vocab_size / 4), int(vocab_size / 6))
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(int(vocab_size / 6), int(vocab_size / 4)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(int(vocab_size / 4), int(vocab_size / 2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(int(vocab_size / 2), int(vocab_size)),
            # torch.nn.Softmax()
        )

        print(f'input: {vocab_size}, output: {self.output_channels}')

        self.last = torch.nn.Linear( int(vocab_size), self.output_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self.last(decoded)
