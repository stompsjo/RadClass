import torch
from torch import nn


class LinearCritic(nn.Module):
    '''
    Largely adapted from a PyTorch conversion of SimCLR by Adam Foster.
    More information found here: https://github.com/ae-foster/pytorch-simclr
    '''

    def __init__(self, latent_dim, temperature=1.):
        super(LinearCritic, self).__init__()
        self.temperature = temperature
        self.projection_dim = 128
        self.w1 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(latent_dim, self.projection_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False)
        self.cossim = nn.CosineSimilarity(dim=-1)

    def project(self, h):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(h)))))

    def forward(self, h1, h2):
        z1, z2 = self.project(h1), self.project(h2)
        sim11 = self.cossim(z1.unsqueeze(-2), z1.unsqueeze(-3)) / self.temperature
        sim22 = self.cossim(z2.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        sim12 = self.cossim(z1.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
        return raw_scores, targets

##########################

'''
Largely adapted from a PyTorch conversion of SimCLR by Andrew Atanov.
More information found here: https://github.com/AndrewAtanov/simclr-pytorch
'''

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import diffdist
import torch.distributed as dist


def gather(z):
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    gather_z = diffdist.functional.all_gather(gather_z, z)
    gather_z = torch.cat(gather_z)

    return gather_z


def accuracy(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    acc = (topk == labels).all(1).float()
    return acc


def mean_cumulative_gain(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    mcg = (topk == labels).float().mean(1)
    return mcg


def mean_average_precision(logits, labels, k):
    # TODO: not the fastest solution but looks fine
    argsort = torch.argsort(logits, dim=1, descending=True)
    labels_to_sorted_idx = torch.sort(torch.gather(torch.argsort(argsort, dim=1), 1, labels), dim=1)[0] + 1
    precision = (1 + torch.arange(k, device=logits.device).float()) / labels_to_sorted_idx
    return precision.sum(1) / k


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc

##########################

import torch.nn as nn
import torchvision

from simclr.modules.resnet_hacks import modify_resnet_model
from simclr.modules.identity import Identity


class SimCLR(nn.Module):
    '''
    Largely adapted from a PyTorch coversion of SimCLR
    found here: https://github.com/Spijkervet/SimCLR
    '''
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j

##########################

import torch.nn.functional as F

class MSELoss(nn.Module):
    """ use just MSE loss with UncertainLinear network """
    def forward(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # yhat, _ = out
        # print('out: {}'.format(out))
        # print('y: {}'.format(y))
        loss = F.mse_loss(out.reshape(-1, 1), y.reshape(-1, 1))
        return loss


class L1Loss(nn.Module):
    """ use just L1 loss with UncertainLinear network """
    def forward(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # yhat, _ = out
        loss = F.smooth_l1_loss(out.reshape(-1, 1), y.reshape(-1, 1))
        return loss
