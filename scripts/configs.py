'''
Author: Jordan Stomps

Largely adapted from a PyTorch conversion of SimCLR by Adam Foster.
More information found here: https://github.com/ae-foster/pytorch-simclr

MIT License

Copyright (c) 2023 Jordan Stomps

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import json

# import torchvision
# import torchvision.transforms as transforms

import sys
sys.path.append('/mnt/palpatine/u9f/RadClass/scripts/')
sys.path.append('/mnt/palpatine/u9f/RadClass/data/')
# from augmentation import ColourDistortion
from dataset import *
from specTools import read_h_file
# from models import *
import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def add_indices(dataset_cls):
    class NewClass(dataset_cls):
        def __getitem__(self, item):
            output = super(NewClass, self).__getitem__(item)
            return (*output, item)

    return NewClass


def get_datasets(dataset, dset_fpath, bckg_fpath, valsfpath=None, testfpath=None, add_indices_to_data=False):# , augment_clf_train=False, num_positive=None):

    # CACHED_MEAN_STD = {
    #     'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     'cifar100': ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
    #     'stl10': ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237)),
    #     'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # }

    # PATHS = {
    #     'minos': '/data/minos/',
    #     'cifar10': '/data/cifar10/',
    #     'cifar100': '/data/cifar100/',
    #     'stl10': '/data/stl10/',
    #     'imagenet': '/data/imagenet/2012/'
    # }
    # try:
    #     with open('dataset-paths.json', 'r') as f:
    #         local_paths = json.load(f)
    #         PATHS.update(local_paths)
    # except FileNotFoundError:
    #     pass
    # root = PATHS[dataset]

    # Data
    # if dataset == 'minos':
    #     img_size = 1000
    # elif dataset == 'stl10':
    #     img_size = 96
    # elif dataset == 'imagenet':
    #     img_size = 224
    # else:
    #     img_size = 32

    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
    #     transforms.RandomHorizontalFlip(),
    #     ColourDistortion(s=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    # ])
    transform_train = [
        transforms.Background(bckg_dir=bckg_fpath, mode='beads'),
        transforms.Resample(),
        transforms.Sig2Bckg(bckg_dir=bckg_fpath, mode='beads', r=(0.5, 1.5)),
        transforms.Nuclear(binE=3),
        transforms.Resolution(multiplier=(0.5, 1.5)),
        transforms.Mask(),
        transforms.GainShift()
    ]

    # if dataset == 'minos':
    #     transform_test = [
    #         transforms.Background(bckg_dir=bckg_fpath, mode='beads'),
    #         transforms.Resample(),
    #         transforms.Sig2Bckg(bckg_dir=bckg_fpath, mode='beads', r=(0.5, 1.5)),
    #         transforms.Nuclear(binE=3),
    #         transforms.Resolution(multiplier=(0.5, 1.5)),
    #         transforms.Mask(),
    #         transforms.GainShift()
    #     ]
    # elif dataset == 'imagenet':
    #     transform_test = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    #     ])
    # else:
    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    #     ])

    # if augment_clf_train:
    #     transform_clftrain = transforms.Compose([
    #         transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    #     ])
    # else:
    #     transform_clftrain = transform_test

    if dataset == 'minos':
        data = pd.read_hdf(dset_fpath, key='data')
        # print(f'\tclasses: {np.unique(targets, return_counts=True)}')
        # print(f'\t\tshape: {targets.shape}')
        ytr = np.full(data.shape[0], -1)
        Xtr = data.to_numpy()[:, np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')
        val = pd.read_hdf(valsfpath, key='data')
        Xval = val.to_numpy()[:, 1+np.arange(1000)].astype(float)
        yval = val['label'].values
        # yval[yval == 1] = 0
        yval[yval != 1] = 0
        test = read_h_file(testfpath, 60, 60)
        Xtest = test.to_numpy()[:, np.arange(1000)].astype(float)
        targets = test['event'].values
        # all test values are positives
        # ytest = np.full_like(ytest, 0, dtype=np.int32)
        ytest = np.ones_like(targets, dtype=np.int32)
        # metal transfers
        ytest[targets == 'ac225'] = 0
        ytest[targets == 'activated-metals'] = 0
        ytest[targets == 'spent-fuel'] = 0
        print(f'\ttraining instances = {Xtr.shape[0]}')
        print(f'\tvalidation instances = {Xval.shape[0]}')
        print(f'\ttest instances = {Xtest.shape[0]}')

        if add_indices_to_data:
            tr_dset = add_indices(MINOSBiaugment(Xtr, ytr, transforms=transform_train))
            val_dset = add_indices(DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std))
            test_dset = add_indices(DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std))
        else:
            tr_dset = MINOSBiaugment(Xtr, ytr, transforms=transform_train)
            val_dset = DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std)
            test_dset = DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std)
    elif dataset == 'minos-curated':
        data = pd.read_hdf(dset_fpath, key='data')
        # print(f'\tclasses: {np.unique(targets, return_counts=True)}')
        # print(f'\t\tshape: {targets.shape}')
        ytr = np.full(data.shape[0], -1)
        Xtr = data.to_numpy()[:, np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')

        test_data = read_h_file(testfpath, 60, 60)
        X = test_data.to_numpy()[:, np.arange(1000)].astype(float)
        y = test_data['event'].values
        Xval, Xtest, val_targets, test_targets = train_test_split(X, y, train_size=0.03, stratify=y)
        # all test values are positives
        # ytest = np.full_like(ytest, 0, dtype=np.int32)
        yval = np.ones_like(val_targets, dtype=np.int32)
        ytest = np.ones_like(test_targets, dtype=np.int32)
        # metal transfers
        yval[val_targets == 'ac225'] = 0
        yval[val_targets == 'activated-metals'] = 0
        yval[val_targets == 'spent-fuel'] = 0
        ytest[test_targets == 'ac225'] = 0
        ytest[test_targets == 'activated-metals'] = 0
        ytest[test_targets == 'spent-fuel'] = 0

        # events = np.unique(data['event'].values)
        # targets = data['event'].replace(events, np.arange(len(events)), inplace=False).values
        # data = data.to_numpy()[:, np.arange(1000)].astype(float)
        # Xtr, Xval, ytr, yval = train_test_split(data, targets, test_size=0.33, stratfy=True)
        print(f'\ttraining instances = {Xtr.shape[0]}')
        print(f'\tvalidation instances = {Xval.shape[0]}')
        print(f'\ttest instances = {Xtest.shape[0]}')

        if add_indices_to_data:
            tr_dset = add_indices(MINOSBiaugment(Xtr, ytr, transforms=transform_train))
            val_dset = add_indices(DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std))
            test_dset = add_indices(DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std))
        else:
            tr_dset = MINOSBiaugment(Xtr, ytr, transforms=transform_train)
            val_dset = DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std)
            test_dset = DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std)
    elif dataset == 'minos-2019':
        ### Including unlabeled spectral data for contrastive learning
        data = pd.read_hdf(dset_fpath, key='data')
        # print(f'\tclasses: {np.unique(targets, return_counts=True)}')
        # print(f'\t\tshape: {targets.shape}')
        ytr = np.full(data.shape[0], -1)
        Xtr = data.to_numpy()[:, np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')

        X = pd.read_hdf(valsfpath, key='data')
        # events = np.unique(X['label'].values)
        y = X['label'].values
        y[y == 1] = 0
        y[y != 0] = 1
        # y = X['event'].replace(events, np.arange(len(events)), inplace=False).values
        X = X.to_numpy()[:, 1+np.arange(1000)].astype(float)
        Xval, Xtest, yval, ytest = train_test_split(X, y, train_size=0.4)
        print(f'\ttraining instances = {Xtr.shape[0]}')
        print(f'\tvalidation instances = {Xval.shape[0]}')
        print(f'\ttest instances = {Xtest.shape[0]}')

        if add_indices_to_data:
            tr_dset = add_indices(MINOSBiaugment(Xtr, ytr, transforms=transform_train))
            val_dset = add_indices(DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std))
            test_dset = add_indices(DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std))
        else:
            tr_dset = MINOSBiaugment(Xtr, ytr, transforms=transform_train)
            val_dset = DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std)
            test_dset = DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std)
    elif dataset == 'minos-2019-binary':
        ### Using only the data that was used for the preliminary experiment
        data = pd.read_hdf(dset_fpath, key='data')
        targets = data['label'].values
        targets[targets == 1] = 0
        targets[targets != 0] = 1
        print(f'\tclasses: {np.unique(targets, return_counts=True)}')
        print(f'\t\tshape: {targets.shape}')
        # targets = data['event'].replace(events, np.arange(len(events)), inplace=False).values
        data = data.to_numpy()[:, 1+np.arange(1000)].astype(float)
        print(f'\tNOTE: double check data indexing: {data.shape}')
        Xtr, X, ytr, y = train_test_split(data, targets, test_size=0.3)
        Xval, Xtest, yval, ytest = train_test_split(X, y, train_size=0.33)
        print(f'\ttraining instances = {Xtr.shape[0]}')
        print(f'\tvalidation instances = {Xval.shape[0]}')
        print(f'\ttest instances = {Xtest.shape[0]}')

        if add_indices_to_data:
            tr_dset = add_indices(MINOSBiaugment(np.append(Xtr, Xval, axis=0), np.append(ytr, yval, axis=0), transforms=transform_train))
            val_dset = add_indices(DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std))
            test_dset = add_indices(DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std))
        else:
            tr_dset = MINOSBiaugment(Xtr, ytr, transforms=transform_train)
            val_dset = DataOrganizer(Xval, yval, tr_dset.mean, tr_dset.std)
            test_dset = DataOrganizer(Xtest, ytest, tr_dset.mean, tr_dset.std)
    # elif dataset == 'cifar100':
    #     if add_indices_to_data:
    #         dset = add_indices(torchvision.datasets.CIFAR100)
    #     else:
    #         dset = torchvision.datasets.CIFAR100
    #     if num_positive is None:
    #         trainset = CIFAR100Biaugment(root=root, train=True, download=True, transform=transform_train)
    #     else:
    #         trainset = CIFAR100Multiaugment(root=root, train=True, download=True, transform=transform_train,
    #                                         n_augmentations=num_positive)
    #     testset = dset(root=root, train=False, download=True, transform=transform_test)
    #     clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
    #     num_classes = 100
    #     stem = StemCIFAR
    # elif dataset == 'cifar10':
    #     if add_indices_to_data:
    #         dset = add_indices(torchvision.datasets.CIFAR10)
    #     else:
    #         dset = torchvision.datasets.CIFAR10
    #     if num_positive is None:
    #         trainset = CIFAR10Biaugment(root=root, train=True, download=True, transform=transform_train)
    #     else:
    #         trainset = CIFAR10Multiaugment(root=root, train=True, download=True, transform=transform_train,
    #                                        n_augmentations=num_positive)
    #     testset = dset(root=root, train=False, download=True, transform=transform_test)
    #     clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
    #     num_classes = 10
    #     stem = StemCIFAR
    # elif dataset == 'stl10':
    #     if add_indices_to_data:
    #         dset = add_indices(torchvision.datasets.STL10)
    #     else:
    #         dset = torchvision.datasets.STl10
    #     if num_positive is None:
    #         trainset = STL10Biaugment(root=root, split='unlabeled', download=True, transform=transform_train)
    #     else:
    #         raise NotImplementedError
    #     testset = dset(root=root, split='train', download=True, transform=transform_test)
    #     clftrainset = dset(root=root, split='test', download=True, transform=transform_clftrain)
    #     num_classes = 10
    #     stem = StemSTL
    # elif dataset == 'imagenet':
    #     if add_indices_to_data:
    #         dset = add_indices(torchvision.datasets.ImageNet)
    #     else:
    #         dset = torchvision.datasets.ImageNet
    #     if num_positive is None:
    #         trainset = ImageNetBiaugment(root=root, split='train', transform=transform_train)
    #     else:
    #         raise NotImplementedError
    #     testset = dset(root=root, split='val', transform=transform_test)
    #     clftrainset = dset(root=root, split='train', transform=transform_clftrain)
    #     num_classes = len(testset.classes)
    #     stem = StemImageNet
    else:
        raise ValueError("Bad dataset value: {}".format(dataset))

    # return trainset, testset, clftrainset, num_classes, stem
    return tr_dset, val_dset, test_dset
