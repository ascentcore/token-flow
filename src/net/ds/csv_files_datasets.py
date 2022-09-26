import os
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset


class CSVFilesDataset(Dataset):
    def __init__(self, folder, batch_size):
        self.file_names = []
        for (dir_path, dir_names, file_names) in os.walk(f'studies/t-ler/data/{cfg.folder}/train'):
            for file_name in file_names:
                if file_name.endswith('.dataset.csv'):
                    self.file_names.append(os.path.join(dir_path, file_name))
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):  # idx means index of the chunk.

        batch_x = self.filenames[idx *
                                 self.batch_size:(idx + 1) * self.batch_size]
        data = []
        labels = []
        label_classes = ["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"]
        for file in batch_x:
            # Change this line to read any other type of file
            temp = pd.read_csv(open(file, 'r'))
            # Convert column data to matrix like data with one channel
            data.append(temp.values.reshape(32, 32, 1))
            # Pattern extracted from file_name
            pattern = "^" + eval("file[14:21]")
            for j in range(len(label_classes)):
                # Pattern is matched against different label_classes
                if re.match(pattern, label_classes[j]):
                    labels.append(j)
        # Because of Pytorch's channel first convention
        data = np.asarray(data).reshape(-1, 1, 32, 32)
        labels = np.asarray(labels)

        # The following condition is actually needed in Pytorch. Otherwise, for our particular example, the iterator will be an infinite loop.
        # Readers can verify this by removing this condition.
        if idx == self.__len__():
            raise IndexError

        return data, labels
