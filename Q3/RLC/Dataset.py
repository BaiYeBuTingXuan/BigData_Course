from abc import ABC

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch


class ModelDataset(Dataset, ABC):
    def __init__(self, data_file, label_file):
        self.data = np.load(data_file, allow_pickle=True)
        self.label = np.load(label_file, allow_pickle=True)
        label_list = []
        for label in self.label:
            if label == 1:
                label_list.append([1.0, 0.0])
            elif label == 0:
                label_list.append([0.0, 1.0])
        self.x = list(zip(self.data, np.array(label_list)))
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx < self.len, "IndexError"
        data = self.x[idx]
        return data


if __name__ == "__main__":
    d = ModelDataset(data_file='./data.npy', label_file='label.npy')
    print(d[0])

