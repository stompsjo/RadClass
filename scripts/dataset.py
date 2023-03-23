import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import logging

def memory_summary():
    # Only import Pympler when we need it. We don't want it to
    # affect our process if we never call memory_summary.
    from pympler import summary, muppy
    mem_summary = summary.summarize(muppy.get_objects())
    rows = summary.format_(mem_summary)
    return '\n'.join(rows)


class DataOrganizer(Dataset):
    def __init__(self, X, y, mean, std):
        self.data = torch.FloatTensor(X.copy())
        self.targets = torch.LongTensor(y.copy())

        self.mean = mean
        self.std = std

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        x -= self.mean
        x = torch.where(self.std == 0., x, x/self.std)
        # x /= self.std

        return x, y


class MINOSBiaugment(Dataset):
    def __init__(self, X, y, transforms):
        # self.data = pd.read_hdf(data_fpath, key='data')
        # self.targets = torch.from_numpy(self.data['event'].values)
        # self.data = torch.from_numpy(self.data[np.arange(1000)].values)
        self.data = torch.FloatTensor(X.copy())
        self.targets = torch.LongTensor(y.copy())
        self.transforms = transforms

        self.mean = torch.mean(self.data, axis=0)
        self.std = torch.std(self.data, axis=0)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        spec, target = self.data[index], self.targets[index]

        if self.transforms is not None:
            aug1, aug2 = np.random.choice(self.transforms, size=2, replace=False)
            logging.debug(f'{index}: aug1={aug1} and aug2={aug2}')
            spec1 = torch.FloatTensor(aug1(spec))
            spec2 = torch.FloatTensor(aug2(spec))

        spec1 -= self.mean
        spec1 = torch.where(self.std == 0., spec1, spec1/self.std)
        # spec1 /= self.std
        spec2 -= self.mean
        spec2 = torch.where(self.std == 0., spec2, spec2/self.std)
        # spec2 /= self.std

        return (spec1, spec2), target, index
