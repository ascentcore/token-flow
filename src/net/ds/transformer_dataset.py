import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    def __init__(self, dataset_file, length=5):
        self.dataset = json.loads(open(dataset_file).read())
        self.length = length

    def __len__(self):
        return len(self.dataset)-self.length

    def __getitem__(self, idx):
        # in_data, out_data = self.dataset[idx]
        input_data = []
        for i in range(idx, idx+self.length):
            in_data, out_data = self.dataset[i]
            input_data.append(in_data)

        output_data = out_data

        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)
