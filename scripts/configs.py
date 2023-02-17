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

# from augmentation import ColourDistortion
from dataset import *
# from models import *
from scripts import transforms
from sklearn.model_selection import train_test_split


def add_indices(dataset_cls):
    class NewClass(dataset_cls):
        def __getitem__(self, item):
            output = super(NewClass, self).__getitem__(item)
            return (*output, item)

    return NewClass


def get_datasets(dataset, dset_fpath, bckg_fpath, add_indices_to_data=False):# , augment_clf_train=False, num_positive=None):

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
        targets = data['event'].values
        data = data[np.arange(1000)].values
        Xtr, Xval, ytr, yval = train_test_split(data, targets, test_size=0.33)

        if add_indices_to_data:
            tr_dset = add_indices(MINOSBiaugment(Xtr, ytr, transforms=transform_train))
            val_dset = add_indices(DataOrganizer(Xval, yval))
        else:
            tr_dset = MINOSBiaugment(Xtr, ytr, transforms=transform_train)
            val_dset = DataOrganizer(Xval, yval)
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
    return tr_dset, val_dset
