#!usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
from pathlib import Path
import os
import random
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
from fastprogress import master_bar, progress_bar
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import OneCycleLR
from pyronear import models
from pyronear.datasets.wildfire import (WildFireDataset, WildFireSplitter, FireLabeler)
from pyronear.datasets.wildfire.split_strategy import *


def set_seed(seed):
    """Set the seed for pseudo-random number generations
    Args:
        seed (int): seed to set for reproducibility
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_batch(model, x, target, optimizer, criterion):
    """Train a model for one iteration
    Args:
        model (torch.nn.Module): model to train
        x (torch.Tensor): input sample
        target (torch.Tensor): output target
        optimizer (torch.optim.Optimizer): parameter optimizer
        criterion (torch.nn.Module): loss used for backpropagation
    Returns:
        batch_loss (float): training loss
    """

    # Forward
    outputs = model(x)

    # Loss computation
    batch_loss = criterion(outputs, target)

    # Backprop
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()


def train_epoch(model, train_loader, optimizer, criterion, master_bar,
                epoch=0, scheduler=None, device='cpu'):
    """Train a model for one epoch
    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training dataloader
        optimizer (torch.optim.Optimizer): parameter optimizer
        criterion (torch.nn.Module): criterion object
        master_bar (fastprogress.MasterBar): master bar of training progress
        epoch (int): current epoch index
        scheduler (torch.optim._LRScheduler, optional): learning rate scheduler
        device (str): device hosting tensor data
    Returns:
        batch_loss (float): latch batch loss
    """

    # Training
    model.train()
    loader_iter = iter(train_loader)
    train_loss = 0
    for _ in progress_bar(range(len(train_loader)), parent=master_bar):

        x, target = next(loader_iter)
        target = target.type(torch.LongTensor).squeeze()
        # Work with tensors on GPU
        if device.startswith('cuda'):
            x, target = x.cuda(), target.cuda()

        batch_loss = train_batch(model, x, target, optimizer, criterion)
        train_loss += batch_loss
        if scheduler:
            scheduler.step()

        master_bar.child.comment = f"Batch loss: {batch_loss:.4}"

    train_loss /= len(train_loader)

    return train_loss


def evaluate(model, test_loader, criterion, device='cpu'):
    """Evaluation a model on a dataloader
    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): validation dataloader
        criterion (torch.nn.Module): criterion object
        device (str): device hosting tensor data
    Returns:
        val_loss (float): validation loss
        acc (float): top1 accuracy
    """
    model.eval()
    val_loss, correct, targets = 0, 0, 0
    with torch.no_grad():
        for x, target in test_loader:

            target = target.type(torch.LongTensor).squeeze()
            # Work with tensors on GPU
            if device.startswith('cuda'):
                x, target = x.cuda(), target.cuda()

            # Forward + Backward & optimize
            outputs = model.forward(x)
            val_loss += criterion(outputs, target).item()
            # Index of max log-probability
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            targets += x.size(0)
    val_loss /= len(test_loader)
    acc = correct / targets

    return val_loss, acc


def main(args):

    if args.deterministic:
        set_seed(42)

    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda:0'
        else:
            args.device = 'cpu'

    #Create Dataloaders

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=args.resize // 7, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=args.resize),  # Image net standards
        transforms.ToTensor(),
        normalize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(size=args.resize // 7),
        transforms.CenterCrop(size=args.resize),
        transforms.ToTensor(),
        normalize
    ])

    tf = {'train': train_transforms, 'test': val_transforms, 'val': val_transforms}

    metadata = pd.read_csv(args.metadata)

    wildfire = WildFireDataset(metadata=metadata, path_to_frames=Path(args.DB), target_names=['fire'])

    ratios = {'train': args.ratio_train, 'val': args.ratio_val, 'test': args.ratio_test}

    splitter = WildFireSplitter(ratios, transforms=tf)
    splitter.fit(wildfire)

    train_loader = DataLoader(splitter.train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(splitter.val, batch_size=args.batch_size, shuffle=True)

    # Model definition
    model = models.__dict__[args.model](imagenet_pretrained=args.pretrained,
                                        num_classes=args.nb_class, lin_features=args.lin_feats,
                                        concat_pool=args.concat_pool, bn_final=args.bn_final,
                                        dropout_prob=args.dropout_prob)

    # Freeze layers
    if not args.unfreeze:
        # Model is sequential
        for p in model[1].parameters():
            p.requires_grad = False

    # Resume
    if args.resume:
        model.load_state_dict(torch.load(args.resume)['model'])

    # Send to device
    model.to(args.device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(),
                           betas=(0.9, 0.99),
                           weight_decay=args.weight_decay)

    # Scheduler
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                 epochs=args.epochs, steps_per_epoch=len(train_loader),
                                                 cycle_momentum=(not isinstance(optimizer, optim.Adam)),
                                                 div_factor=args.div_factor, final_div_factor=args.final_div_factor)

    best_loss = math.inf
    mb = master_bar(range(args.epochs))
    for epoch_idx in mb:
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 master_bar=mb, epoch=epoch_idx, scheduler=lr_scheduler,
                                 device=args.device)

        # Evaluation
        val_loss, acc = evaluate(model, val_loader, criterion, device=args.device)

        mb.comment = f"Epoch {epoch_idx+1}/{args.epochs}"
        mb.write(f"Epoch {epoch_idx+1}/{args.epochs} - Training loss: {train_loss:.4} | "
                 f"Validation loss: {val_loss:.4} | Error rate: {1 - acc:.4}")

        # State saving
        if val_loss < best_loss:
            if args.output_dir:
                print(f"Validation loss decreased {best_loss:.4} --> {val_loss:.4}: saving state...")
                torch.save(dict(model=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                lr_scheduler=lr_scheduler.state_dict(),
                                epoch=epoch_idx,
                                args=args),
                           Path(args.output_dir, f"{args.checkpoint}.pth"))
            best_loss = val_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyroNear Classification Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input / Output
    parser.add_argument('--DB', default='./DB', help='dataset root folder')
    parser.add_argument('--metadata', default='./DB/metadata.csv', help='Paht to metadata')
    parser.add_argument('--resume', default=None, help='checkpoint file to resume from')
    parser.add_argument('--output_dir', default=None, help='path for output saving')
    parser.add_argument('--checkpoint', default=None, type=str, help='name of output file')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch index')
    # Architecture
    parser.add_argument('--model', default='resnet18', type=str, help='model architecture')
    parser.add_argument('--nb-class', default=2, type=int, help='train ratio')
    parser.add_argument("--concat_pool", dest="concat_pool",
                        help="replaces AdaptiveAvgPool2d with AdaptiveConcatPool2d",
                        action="store_true")
    parser.add_argument('--lin-feats', default=512, type=int,
                        help='number of nodes in intermediate head layers')
    parser.add_argument("--bn-final", dest="bn_final",
                        help="adds a batch norm layer after last FC",
                        action="store_true")
    parser.add_argument('--dropout-prob', default=0.5, type=float, help='dropout rate of last FC layer')
    parser.add_argument("--binary", dest="binary",
                        help="should the task be considered as binary Classification",
                        action="store_true")
    parser.add_argument("--pretrained", dest="pretrained",
                        help="use ImageNet pre-trained parameters",
                        action="store_true")
    # Device
    parser.add_argument('--device', default=None, help='device')
    parser.add_argument("--deterministic", dest="deterministic",
                        help="should the training be performed in deterministic mode",
                        action="store_true")
    # Loader
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('-s', '--resize', default=224, type=int, help='image size after resizing')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--ratio-train', default=0.8, type=float, help='train ratio')
    parser.add_argument('--ratio-val', default=0.2, type=float, help='validation ratio')
    parser.add_argument('--ratio-test', default=0, type=float, help='test ratio')
    # Optimizer
    parser.add_argument('--lr', default=3e-4, type=float, help='maximum learning rate')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument("--unfreeze", dest="unfreeze", help="should all layers be unfrozen",
                        action="store_true")
    # Scheduler
    parser.add_argument('--div-factor', default=25., type=float,
                        help='div factor of OneCycle policy')
    parser.add_argument('--final-div-factor', default=1e4, type=float,
                        help='final div factor of OneCycle policy')
    args = parser.parse_args()

    main(args)
