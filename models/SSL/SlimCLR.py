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

'''Train an encoder using Contrastive Learning.'''
import argparse
import os
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
# from torchlars import LARS
from tqdm import tqdm

from scripts.configs import get_datasets
from models.PyTorch.critic import LinearCritic
from scripts.evaluate import save_checkpoint, encode_train_set, train_clf, test
from models import *
from scripts.scheduler import CosineAnnealingWithLinearRampLR

from models.PyTorch.ann import LinearNN

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning.')
parser.add_argument('--base-lr', default=0.25, type=float, help='base learning rate, rescaled by batch_size/256')
parser.add_argument("--momentum", default=0.9, type=float, help='SGD momentum')
parser.add_argument('--resume', '-r', type=str, default='', help='resume from checkpoint with this filename')
parser.add_argument('--dataset', '-d', type=str, default='minos', help='dataset keyword',
                    choices=['minos', 'cifar10', 'cifar100', 'stl10', 'imagenet'])
parser.add_argument('--dfpath', '-p', type=str, help='filepath for dataset')
parser.add_argument('--bfpath', '-f', type=str, help='filepath for background library augmentations')
parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
parser.add_argument("--num-epochs", type=int, default=100, help='Number of training epochs')
parser.add_argument("--cosine-anneal", action='store_true', help="Use cosine annealing on the learning rate")
parser.add_argument("--arch", type=str, default='minos', help='Encoder architecture',
                    choices=['minos', 'resnet18', 'resnet34', 'resnet50'])
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--test-freq", type=int, default=10, help='Frequency to fit a linear clf with L-BFGS for testing'
                                                              'Not appropriate for large datasets. Set 0 to avoid '
                                                              'classifier only training here.')
parser.add_argument("--filename", type=str, default='ckpt.pth', help='Output file name')
parser.add_argument('--in-dim', '-i', type=int, help='number of input image dimensions')
parser.add_argument('--mid', '-m', type=int, help='hidden layer size')
parser.add_argument('--n-layers', '-n', type=int, help='number of hidden layers')
args = parser.parse_args()
args.lr = args.base_lr * (args.batch_size / 256)

args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
args.git_diff = subprocess.check_output(['git', 'diff'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
clf = None

print('==> Preparing data..')
# trainset, testset, clftrainset, num_classes, stem = get_datasets(args.dataset, args.dfpath, args.bfpath)
num_classes = 7
trainset, valset = get_datasets(args.dataset, args.dfpath, args.bfpath)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, pin_memory=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
#                                          pin_memory=True)
# clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                            #  pin_memory=True)

# Model
print('==> Building model..')
##############################################################
# Encoder
##############################################################
# if args.arch == 'resnet18':
#     net = ResNet18(stem=stem)
# elif args.arch == 'resnet34':
#     net = ResNet34(stem=stem)
# elif args.arch == 'resnet50':
#     net = ResNet50(stem=stem)
if args.arch == 'minos':
    net = LinearNN(dim=args.in_dim, mid=args.mid,
                   n_layers=args.n_layers, dropout_rate=1.,
                   n_epochs=args.num_epochs, mid_bias=True,
                   out_bias=True, n_classes=None)
else:
    raise ValueError("Bad architecture specification")
net = net.to(device)

##############################################################
# Critic
##############################################################
critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

if device == 'cuda':
    repr_dim = net.representation_dim
    net = torch.nn.DataParallel(net)
    net.representation_dim = repr_dim
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    resume_from = os.path.join('./checkpoint', args.resume)
    checkpoint = torch.load(resume_from)
    net.load_state_dict(checkpoint['net'])
    critic.load_state_dict(checkpoint['critic'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
base_optimizer = optim.SGD(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6,
                           momentum=args.momentum)
if args.cosine_anneal:
    scheduler = CosineAnnealingWithLinearRampLR(base_optimizer, args.num_epochs)
# encoder_optimizer = LARS(base_optimizer, trust_coef=1e-3)
encoder_optimizer = base_optimizer


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    critic.train()
    train_loss = 0
    t = tqdm(enumerate(trainloader), desc='Loss: **** ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, _, _) in t:
        x1, x2 = inputs
        x1, x2 = x1.to(device), x2.to(device)
        encoder_optimizer.zero_grad()
        representation1, representation2 = net(x1), net(x2)
        raw_scores, pseudotargets = critic(representation1, representation2)
        loss = criterion(raw_scores, pseudotargets)
        loss.backward()
        encoder_optimizer.step()

        train_loss += loss.item()

        t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))


for epoch in range(start_epoch, start_epoch + args.num_epochs):
    train(epoch)
    if (args.test_freq > 0) and (epoch % args.test_freq == (args.test_freq - 1)):
        X, y = encode_train_set(valloader, device, net)
        clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=1e-5)
        acc = test(valloader, device, net, clf)
        if acc > best_acc:
            best_acc = acc
        save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__))
    elif args.test_freq == 0:
        save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__))
    if args.cosine_anneal:
        scheduler.step()
