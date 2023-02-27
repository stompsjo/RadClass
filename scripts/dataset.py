import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
        x = torch.where(self.std == 0., 0., x/self.std)
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
        # print(f'mean={self.mean}\nstd={self.std}')

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
            spec1 = torch.FloatTensor(aug1(spec))
            spec2 = torch.FloatTensor(aug2(spec))

        # print(f'spec1 shape={spec1.shape}, mean shape={self.mean.shape}, and std shape={self.std.shape}')
        spec1 -= self.mean
        spec1 = torch.where(self.std == 0., 0., spec1/self.std)
        # spec1 /= self.std
        spec2 -= self.mean
        spec2 = torch.where(self.std == 0., 0., spec2/self.std)
        # spec2 /= self.std

        # print(f'spec1={spec1[:10]}\nspec2={spec2[:10]}')

        return (spec1, spec2), target, index
