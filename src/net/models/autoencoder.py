import torch
import torch.nn.functional as F


class AE(torch.nn.Module):
    def __init__(self, vocab_size, input_channels = 1, output_channels=1, steps = 8):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        vocab_size = vocab_size * input_channels

        layers = [torch.nn.Flatten()]

        for i in range(steps):
            layers.append(torch.nn.Linear(vocab_size, int(vocab_size * 0.9)))
            layers.append(torch.nn.ReLU())
            vocab_size = int(vocab_size * 0.9)


        print(f'Last size: {vocab_size}')
        print(layers)
        self.encoder = torch.nn.Sequential(*layers)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(int(vocab_size / 6), int(vocab_size / 4)),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(int(vocab_size / 4), int(vocab_size / 2)),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(int(vocab_size / 2), int(vocab_size)),
            # torch.nn.Softmax()
        )

        print(f'input: {vocab_size}, output: {self.output_channels}')

        self.last = torch.nn.Linear(int(vocab_size), self.output_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        return self.last(encoded)
