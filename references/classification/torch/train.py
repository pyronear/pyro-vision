#!usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from pathlib import Path
import math
import torch
import torch.utils.data
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
from fastprogress import master_bar, progress_bar

from pyronear.datasets import OpenFire

# Disable warnings about RGBA images (discard transparency information)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")


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
        loader_iter (iter(torch.utils.data.DataLoader)): training dataloader iterator
        optimizer (torch.optim.Optimizer): parameter optimizer
        criterion (torch.nn.Module): criterion object
    Returns:
        batch_loss (float): training loss
        acc (float): top1 accuracy on batch
    """

    # Forward
    outputs = model(x)

    # Loss computation
    batch_loss = criterion(outputs, target)
    # Topk predictions
    pred = outputs.topk(1, dim=1)[1]
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    acc = correct[:, 0].sum().item() / target.size(0)

    # Backprop
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item(), acc


def train_epoch(model, train_loader, optimizer, criterion, master_bar,
                epoch=0, scheduler=None):
    """Train a model for one epoch
    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training dataloader
        optimizer (torch.optim.Optimizer): parameter optimizer
        criterion (torch.nn.Module): criterion object
        master_bar (fastprogress.MasterBar): master bar of training progress
        epoch (int): current epoch index
    """

    # Training
    model.train()
    loader_iter = iter(train_loader)
    running_loss = 0
    for batch_idx in progress_bar(range(len(train_loader)), parent=master_bar):

        x, target = next(loader_iter)
        if torch.cuda.is_available():
            x, target = x.cuda(non_blocking=True), target.cuda(non_blocking=True)

        batch_loss, acc = train_batch(model, x, target, optimizer, criterion)
        if scheduler:
            scheduler.step()

        master_bar.child.comment = f"Training loss: {batch_loss:.4} (Acc@1: {acc:.2%})"


def evaluate(model, test_loader, criterion):
    """Evaluation a model on a dataloader
    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): validation dataloader
        criterion (torch.nn.Module): criterion object
    Returns:
        val_loss (float): validation loss
        acc (float): top1 accuracy
    """
    model.eval()
    val_loss, correct, targets = 0, 0, 0
    with torch.no_grad():
        for x, target in test_loader:
            # Work with tensors on GPU
            if torch.cuda.is_available():
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((args.resize, args.resize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        normalize
    ])

    # Train & test sets
    train_set = OpenFire(root=args.data_path, train=True, download=True, valid_pct=0.2,
                         transform=train_transforms)
    val_set = OpenFire(root=args.data_path, train=False, download=True, valid_pct=0.2,
                       transform=test_transforms)
    num_classes = len(train_set.classes)
    # Samplers
    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(val_set)

    # Data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                             num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, sampler=test_sampler,
                                            num_workers=args.workers, pin_memory=True)

    # Model definition
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)

    # Change fc
    in_features = getattr(model, 'fc').in_features
    setattr(model, 'fc', nn.Linear(in_features, num_classes))
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
    best_acc = 0
    mb = master_bar(range(args.epochs))
    for epoch_idx in mb:
        # Training
        train_epoch(model, train_loader, optimizer, criterion, mb, epoch=epoch_idx, scheduler=lr_scheduler)

        # Evaluation
        val_loss, acc = evaluate(model, test_loader, criterion)

        mb.first_bar.comment = f"Epoch {epoch_idx+1}/{args.epochs}"
        mb.write(f'Epoch {epoch_idx+1}/{args.epochs} - Validation loss: {val_loss:.4} (Acc@1: {acc:.2%})')

        # State saving
        if val_loss < best_loss:
            print(f"Validation loss decreased {best_loss:.4} --> {val_loss:.4}: saving state...")
            best_loss = val_loss
            best_acc = acc
            if args.output_dir:
                torch.save(dict(model=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                lr_scheduler=lr_scheduler.state_dict(),
                                epoch=epoch_idx,
                                args=args),
                           Path(args.output_dir, f"{args.checkpoint}.pth"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyroNear Classification Training')
    parser.add_argument('--data-path', default='./tmp_data', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-s', '--resize', default=224, type=int)
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')
    parser.add_argument('--div-factor', default=25., type=float, help='div factor of OneCycle policy')
    parser.add_argument('--final-div-factor', default=1e4, type=float, help='final div factor of OneCycle policy')
    parser.add_argument('--output-dir', default=None, help='path where to save')
    parser.add_argument('--checkpoint', default='checkpoint', type=str, help='name of output file')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        help="Should the training be performed in deterministic mode",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
