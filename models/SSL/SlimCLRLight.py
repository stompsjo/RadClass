import argparse
import os
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
# from torchlars import LARS
from tqdm import tqdm

import sys
sys.path.append('/mnt/palpatine/u9f/RadClass/scripts/')
sys.path.append('/mnt/palpatine/u9f/RadClass/models/PyTorch/')
sys.path.append('/mnt/palpatine/u9f/RadClass/models/SSL/')

from configs import get_datasets
from critic import LinearCritic
from lightModel import LitSimCLR
from evaluate import save_checkpoint, encode_train_set, train_clf, test
# from models import *
from scheduler import CosineAnnealingWithLinearRampLR
from ann import LinearNN

from pytorch_metric_learning.losses import SelfSupervisedLoss, NTXentLoss
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

import numpy as np
import joblib

import logging
import copy

# needed for lightning's distributed package
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
# torch.distributed.init_process_group("gloo")

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


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning.')
    parser.add_argument('--base-lr', default=0.25, type=float,
                        help='base learning rate, rescaled by batch_size/256')
    parser.add_argument("--momentum", default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--resume', '-r', type=str, default='',
                        help='resume from checkpoint with this filename')
    parser.add_argument('--dataset', '-d', type=str, default='minos',
                        help='dataset keyword',
                        choices=['minos', 'minos-ssml', 'minos-transfer-ssml', 'minos-curated',
                                 'minos-2019', 'minos-2019-binary',
                                 'cifar10', 'cifar100', 'stl10', 'imagenet'])
    parser.add_argument('--dfpath', '-p', type=str,
                        help='filepath for dataset')
    parser.add_argument('--valfpath', '-v', type=str,
                        help='filepath for validation dataset')
    parser.add_argument('--testfpath', '-t', type=str,
                        help='filepath for test dataset')
    parser.add_argument('--bfpath', '-f', type=str,
                        help='filepath for background library augmentations')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='InfoNCE temperature')
    parser.add_argument("--batch-size", type=int, default=512,
                        help='Training batch size')
    parser.add_argument("--num-epochs", type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument("--cosine-anneal", action='store_true',
                        help="Use cosine annealing on the learning rate")
    parser.add_argument("--normalization", action='store_true',
                        help="Use normalization instead of standardization in pre-processing.")
    parser.add_argument("--accounting", action='store_true',
                        help='Remove estimated background before returning spectra in training.')
    parser.add_argument("--arch", type=str, default='minos',
                        help='Encoder architecture',
                        choices=['minos', 'minos-ssml', 'minos-transfer-ssml', 'minos-curated', 'minos-2019',
                                 'minos-2019-binary', 'resnet18', 'resnet34', 'resnet50'])
    parser.add_argument("--num-workers", type=int, default=2,
                        help='Number of threads for data loaders')
    parser.add_argument("--test-freq", type=int, default=10,
                        help='Frequency to fit a linear clf with L-BFGS for testing'
                             'Not appropriate for large datasets. Set 0 to avoid '
                             'classifier only training here.')
    parser.add_argument("--filename", type=str, default='ckpt',
                        help='Output file name')
    parser.add_argument('--in-dim', '-i', type=int,
                        help='number of input image dimensions')
    parser.add_argument('--mid', '-m', type=int, nargs='+',
                        help='hidden layer size')
    parser.add_argument('--n-layers', '-n', type=int,
                        help='number of hidden layers')
    parser.add_argument('--n-classes', '-c', type=int, default=7,
                        help='number of classes/labels in projection head')
    parser.add_argument('--alpha', '-a', type=float, default=0.999,
                        help='weight for semi-supervised contrastive loss')

    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(filename='debug.log',
                        filemode='a',
                        level=logging.INFO)
    args = parse_arguments()
    args.lr = args.base_lr * (args.batch_size / 256)

    args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    args.git_diff = subprocess.check_output(['git', 'diff'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # for use with a GPU
    if device == 'cuda':
        torch.set_float32_matmul_precision('medium')
    print(f'device used={device}')
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    clf = None

    # set seed(s) for reproducibility
    torch.manual_seed(20230316)
    np.random.seed(20230316)

    print('==> Preparing data..')
    print('min-max normalization? ', args.normalization)
    num_classes = args.n_classes
    trainset, valset, testset, ssmlset = get_datasets(args.dataset, args.dfpath,
                                                      args.bfpath, args.valfpath,
                                                      args.testfpath, args.normalization,
                                                      args.accounting)

    pin_memory = True if device == 'cuda' else False
    print(f'pin_memory={pin_memory}')

    if ssmlset is not None:
        full_trainset = torch.utils.data.ConcatDataset([trainset, ssmlset])
    else:
        full_trainset = trainset
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=pin_memory)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=pin_memory)
    # if ssmlset is not None:
    #     ssmlloader = torch.utils.data.DataLoader(ssmlset,
    #                                              batch_size=args.batch_size,
    #                                              shuffle=True,
    #                                              num_workers=args.num_workers,
    #                                              pin_memory=pin_memory)
    #     to_model = [trainloader, ssmlloader]
    # else:
    #     to_model = [trainloader]
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=pin_memory)

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
    if args.arch in ['minos', 'minos-ssml', 'minos-transfer-ssml', 'minos-curated', 'minos-2019', 'minos-2019-binary']:
        net = LinearNN(dim=args.in_dim, mid=args.mid,
                       n_layers=args.n_layers, dropout_rate=1.,
                       n_epochs=args.num_epochs, mid_bias=True,
                       out_bias=True, n_classes=None)
    else:
        raise ValueError("Bad architecture specification")
    net = net.to(device)
    print(f'net dimensions={net.representation_dim}')

    ##############################################################
    # Critic
    ##############################################################
    # projection head to reduce dimensionality for contrastive loss
    proj_head = LinearCritic(latent_dim=args.mid[-1]).to(device)
    # classifier for better decision boundaries
    # latent_clf = nn.Linear(proj_head.projection_dim, num_classes).to(device)
    # NTXentLoss on its own requires labels (all unique)
    critic = NTXentLoss(temperature=0.07, reducer=reducers.DoNothingReducer())
    sub_batch_size = 64

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

    # base_optimizer = optim.SGD(list(net.parameters()) + list(proj_head.parameters())
    #                            + list(latent_clf.parameters()) + list(critic.parameters()),
    #                            lr=args.lr, weight_decay=1e-6, momentum=args.momentum)
    # base_optimizer = optim.SGD(list(net.parameters()) + list(proj_head.parameters())
    #                            + list(critic.parameters()),
    #                            lr=args.lr, weight_decay=1e-6, momentum=args.momentum)
    # if args.cosine_anneal:
    #     scheduler = CosineAnnealingWithLinearRampLR(base_optimizer, args.num_epochs)
    # # encoder_optimizer = LARS(base_optimizer, trust_coef=1e-3)
    # encoder_optimizer = base_optimizer

    # make checkpoint directory
    ckpt_path = './checkpoint/'+args.filename+'/'
    if not os.path.isdir(ckpt_path): os.mkdir(ckpt_path)

    # save statistical data
    joblib.dump(trainset.mean, ckpt_path+args.filename+'-train_means.joblib')
    joblib.dump(trainset.std, ckpt_path+args.filename+'-train_stds.joblib')

    lightning_model = LitSimCLR(net, proj_head, critic, args.batch_size, sub_batch_size, args.lr, args.momentum, args.cosine_anneal, args.num_epochs, args.alpha, num_classes, args.test_freq, testloader)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=ckpt_path)
    trainer = pl.Trainer(max_epochs=args.num_epochs, default_root_dir=ckpt_path, check_val_every_n_epoch=args.test_freq, profiler='simple', limit_train_batches=0.75, logger=tb_logger, num_sanity_val_steps=0)
    trainer.fit(model=lightning_model, train_dataloaders=trainloader, val_dataloaders=valloader)
    trainer.test(model=lightning_model, dataloaders=testloader)



    # bacc_curve = np.array([])
    # train_loss_curve = np.array([])
    # test_loss_curve = np.array([])
    # confmat_curve = np.array([])
    # with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(
    #         wait=2,
    #         warmup=2,
    #         active=6,
    #         repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler,
    #     with_stack=True
    # ) as profiler:
    #     for epoch in range(start_epoch, start_epoch + args.num_epochs):
    #         train_loss = train(epoch)
    #         train_loss_curve = np.append(train_loss_curve, train_loss)
    #         if (args.test_freq > 0) and (epoch % args.test_freq == (args.test_freq - 1)):
    #             X, y = encode_train_set(valloader, device, net)
    #             clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=1e-5)
    #             acc, bacc, cmat, test_loss = test(testloader, device, net, clf, num_classes)
    #             bacc_curve = np.append(bacc_curve, bacc)
    #             test_loss_curve = np.append(test_loss_curve, test_loss)
    #             confmat_curve = np.append(confmat_curve, cmat)
    #             print(f'\t-> epoch {epoch} Balanced Accuracy = {bacc}')
    #             print(f'\t-> with confusion matrix = {cmat}')
    #             if acc > best_acc:
    #                 best_acc = acc
    #             # save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__))
    #             save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__))
    #             results = {'bacc_curve': bacc_curve, 'train_loss_curve': train_loss_curve,
    #                     'test_loss_curve': test_loss_curve, 'confmat_curve': confmat_curve}
    #             joblib.dump(results, './checkpoint/'+args.filename+'-result_curves.joblib')
    #         elif args.test_freq == 0:
    #             # save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__))
    #             save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__))
    #         if args.cosine_anneal:
    #             scheduler.step()


if __name__ == "__main__":
    main()
