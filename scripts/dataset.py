import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DataOrganizer(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x.copy())
        self.y = torch.FloatTensor(y.copy())

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx


class MINOSBiaugment(Dataset):
    def __init__(self, X, y, transforms):
        # self.data = pd.read_hdf(data_fpath, key='data')
        # self.targets = torch.from_numpy(self.data['event'].values)
        # self.data = torch.from_numpy(self.data[np.arange(1000)].values)
        self.data = torch.FloatTensor(X.copy())
        self.targets = torch.FloatTensor(y.copy())
        self.transforms = transforms

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        spec, target = self.data.iloc[index].to_numpy().astype(float), self.targets.iloc[index]

        if self.transform is not None:
            aug1, aug2 = np.random.choice(self.transforms, size=2, replace=False)
            spec1 = aug1(spec)
            spec2 = aug2(spec)

        return (spec1, spec2), target, index
