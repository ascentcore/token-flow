import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset


class LinearDataset(Dataset):
    def __init__(self, dataset_file):
        self.dataset = json.loads(open(dataset_file).read())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        in_data, out_data = self.dataset[idx]

        input = torch.tensor(in_data, dtype=torch.float32)
        output = torch.tensor(out_data, dtype=torch.float32)

        return input, output
