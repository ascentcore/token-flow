import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset


class LinearRuntimeDataset(Dataset):
    def __init__(self, dataset):
        self.inner_ds = []
        for ds in dataset.datasets.values():
            self.inner_ds.extend(ds.data)

    def __len__(self):
        return len(self.inner_ds)

    def __getitem__(self, idx):
        in_data, out_data = self.inner_ds[idx]

        input = torch.tensor(in_data, dtype=torch.float32)
        output = torch.tensor(out_data, dtype=torch.float32)

        return input, output
