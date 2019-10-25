#!usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial

import torch
from torch import nn
import warnings
from fastai.torch_core import defaults
from fastai import vision
from fastai.data_block import CategoryList, FloatList
from pyronear.datasets import OpenFire

np.random.seed(42)
# Disable warnings from fastai using deprecated functions for PyTorch>=1.3
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


class CustomBCELogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, x, target):
        # Reshape output tensor for BCELoss
        return super(CustomBCELogitsLoss, self).forward(x, target.view(-1, 1))


def main(args):

    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda:0'
        else:
            args.device = 'cpu'

    defaults.device = torch.device(args.device)

    # Aggregate path and labels into list for fastai ImageDataBunch
    fnames, labels, is_valid = [], [], []
    dataset = OpenFire(root=args.data_path, train=True, download=True)
    for sample in dataset.data:
        fnames.append(dataset._images.joinpath(sample['name']).relative_to(dataset.root))
        labels.append(sample['target'])
        is_valid.append(False)
    dataset = OpenFire(root=args.data_path, train=False, download=True)
    for sample in dataset.data:
        fnames.append(dataset._images.joinpath(sample['name']).relative_to(dataset.root))
        labels.append(sample['target'])
        is_valid.append(True)

    df = pd.DataFrame.from_dict(dict(name=fnames, label=labels, is_valid=is_valid))

    # Split train and valid sets
    il = vision.ImageList.from_df(df, path=args.data_path).split_from_df('is_valid')
    # Encode labels
    il = il.label_from_df(cols='label', label_cls=FloatList if args.binary else CategoryList)
    # Set transformations
    il = il.transform(vision.get_transforms(), size=args.resize)
    # Create the Databunch
    data = il.databunch(bs=args.batch_size, num_workers=args.workers).normalize(vision.imagenet_stats)
    # Metric
    metric = partial(vision.accuracy_thresh, thresh=0.5) if args.binary else vision.error_rate

    learner = vision.cnn_learner(data, vision.models.__dict__[args.model],
                                 pretrained=args.pretrained,
                                 wd=args.weight_decay,
                                 ps=args.dropout_prob,
                                 concat_pool=args.concat_pool,
                                 loss_func=CustomBCELogitsLoss() if args.binary else nn.CrossEntropyLoss(),
                                 metrics=metric)

    if args.resume:
        learner.load(args.resume)
    if args.unfreeze:
        learner.unfreeze()

    learner.fit_one_cycle(args.epochs, max_lr=slice(None, args.lr, None),
                          div_factor=args.div_factor)

    learner.save(args.checkpoint)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyroNear Classification Training with Fastai')
    parser.add_argument('--data-path', default='./data', help='dataset')
    parser.add_argument('--model', default='resnet18', type=str, help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('-s', '--resize', default=224, type=int)
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=3e-3, type=float, help='initial learning rate')
    parser.add_argument("--concat-pool", dest="concat_pool",
                        help="Use pre-trained models from the modelzoo",
                        action="store_true")
    parser.add_argument('--dropout-prob', default=0.5, type=float, help='dropout rate of last FC layer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')
    parser.add_argument('--div-factor', default=25., type=float, help='div factor of OneCycle policy')
    parser.add_argument('--checkpoint', default='checkpoint', type=str, help='name of output file')
    parser.add_argument('--resume', default=None, help='checkpoint name to resume from (default: None)')
    parser.add_argument("--binary", dest="binary", help="Should the task be considered as binary Classification",
                        action="store_true")
    parser.add_argument("--unfreeze", dest="unfreeze", help="Should all layers be unfrozen",
                        action="store_true")
    parser.add_argument("--pretrained", dest="pretrained",
                        help="Use pre-trained models from the modelzoo",
                        action="store_true")
    args = parser.parse_args()

    main(args)
