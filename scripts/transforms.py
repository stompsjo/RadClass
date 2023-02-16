from scripts.augs import DANSE
import numpy as np
import pandas as pd
from scipy.stats import loguniform
import torch


class Background(torch.nn.Module):
    def __init__(self, bckg_dir, mode='beads'):
        super().__init__()
        # _log_api_usage_once(self)

        self.mode = mode
        # bckg_man = pd.read_hdf('/home/u9f/Documents/manual_background_spectra.hdf5', key='data')
        # bckg_bds = pd.read_hdf('/home/u9f/Documents/beads_background_spectra.hdf', key='data')
        self.bckg = pd.read_hdf(bckg_dir, mode='beads')
        self.bckg_dir = bckg_dir

    def forward(self, X):
        X = X.copy()
        bckg_idx = np.random.choice(self.bckg.shape[0])
        ibckg = self.bckg.iloc[bckg_idx][np.arange(1000)].to_numpy().astype(float)
        auger = DANSE()
        return auger.background(X, ibckg, subtraction=True, event_idx=None, mode='beads')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bckg_dir={self.bckg_dir}, mode={self.mode})"


class Resample(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X = X.copy()
        auger = DANSE()
        return auger.resample(X)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class Sig2Bckg(torch.nn.Module):
    def __init__(self, bckg_dir, mode='beads', r=(0.5, 2.)):
        super().__init__()
        # _log_api_usage_once(self)

        self.mode = mode
        # bckg_man = pd.read_hdf('/home/u9f/Documents/manual_background_spectra.hdf5', key='data')
        # bckg_bds = pd.read_hdf('/home/u9f/Documents/beads_background_spectra.hdf', key='data')
        self.bckg = pd.read_hdf(bckg_dir, mode='beads')
        self.bckg_dir = bckg_dir
        self.r = r

    def forward(self, X):
        X = X.copy()
        bckg_idx = np.random.choice(self.bckg.shape[0])
        ibckg = self.bckg.iloc[bckg_idx][np.arange(1000)].to_numpy().astype(float)
        auger = DANSE()
        return auger.sig2bckg(X, ibckg, r=self.r, subtraction=True, event_idx=None, mode='beads')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bckg_dir={self.bckg_dir}, mode={self.mode}, r={self.r})"


class Nuclear(torch.nn.Module):
    def __init__(self, binE=3.):
        super().__init__()

        self.binE = binE

    def forward(self, X):
        X = X.copy()
        nuclides = {'K40Th232': [1460, 2614], 'U238': [609], 'Bi214': [1764, 2204], 'Pb214': [295, 352], 'Ar41': [1294]}
        nkey = np.random.choice(np.array(list(nuclides.keys)))
        for e in nuclides[nkey]:
            chE = e/self.binE
            roi = [int(max(chE-int(len(X)*0.01), 0)), int(min(chE+int(len(X)*0.01), len(X)-1))]
            auger = DANSE()
            try:
                X = auger.nuclear(roi, X, escape=False, binE=self.binE, subtract=False)
            except (RuntimeError, IndexError) as e: # ignore unsuccessful peak fits
                continue
        return X

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(binE={self.binE})"


class Resolution(torch.nn.Module):
    def __init__(self, multiplier=(0.5, 1.5)):
        super().__init__()

        if multiplier[0] <= 0 or multiplier[1] <= 0:
            raise ValueError('{} must be positive.'.format(multiplier))
        self.multiplier = multiplier

    def forward(self, X):
        X = X.copy()
        auger = DANSE()
        sucess = False
        while not sucess:
            roi = auger.find_res(X)
            multiplier = loguniform(multiplier[0], multiplier[1], size=1)
            conserve = np.random.choice([True, False])
            try:
                X = auger.resolution(roi, X.copy(), multiplier=multiplier, conserve=conserve)
                success = True
            except (RuntimeError, IndexError) as e: # ignore unsuccessful peak fits
                sucess = False
                continue
        return X

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(multiplier={self.multiplier})"


class Mask(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X = X.copy()
        auger = DANSE()
        return auger.mask(X, mode='block', block=(0, np.random.randint(20, 100)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class GainShift(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X = X.copy()
        auger = DANSE()

        k = np.random.randint(-5, 5)
        lam = np.random.uniform(-5, 5)
        return auger.gain_shift(X, bins=None, lam=lam, k=k, mode='resample')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
