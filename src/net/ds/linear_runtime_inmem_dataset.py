import torch
from torch.utils.data import Dataset


class LinearRuntimeInMemDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, idx):
        in_data, out_data = self.dataset.data[idx]


        input = torch.tensor(in_data, dtype=torch.float32)
        output = torch.tensor(out_data, dtype=torch.float32)

        return input, output
