import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset


class TransformerRuntimeDataset(Dataset):
    def __init__(self, dataset, length=10):
        self.length = length
        self.inner_ds = []
        for ds in dataset.datasets.values():
            self.inner_ds.extend(ds.data)

    def __len__(self):
        return len(self.inner_ds)-self.length

    def __getitem__(self, idx):
        # in_data, out_data = self.dataset[idx]
        input_data = []
        for i in range(idx, idx+self.length):
            in_data, out_data = self.inner_ds[i]
            input_data.append(in_data)

        output_data = out_data

        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)
