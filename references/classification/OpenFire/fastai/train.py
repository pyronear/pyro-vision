#!usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) The pyronear developers.
# This file is dual licensed under the terms of the CeCILL-2.1 and GPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import random
import os
import numpy as np
import pandas as pd
from functools import partial

import torch
from torch import nn
import warnings
from fastai.torch_core import defaults
from fastai import vision
from fastai.data_block import CategoryList, FloatList
from fastai.basic_train import Learner
from fastai.vision.learner import model_meta, _default_meta

from pyronear.datasets import OpenFire
from pyronear import models


# Disable warnings from fastai using deprecated functions for PyTorch>=1.3
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# Add split meta data since fastai doesn't have mobilenet
model_meta[models.mobilenet_v2] = lambda m: (m[0][17], m[1])


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


class CustomBCELogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, x, target):
        # Reshape output tensor for BCELoss
        return super(CustomBCELogitsLoss, self).forward(x, target.view(-1, 1))


def main(args):

    if args.deterministic:
        set_seed(42)

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
    # Create model
    model = models.__dict__[args.model](imagenet_pretrained=args.pretrained,
                                        num_classes=data.c, lin_features=args.lin_feats,
                                        concat_pool=args.concat_pool, bn_final=args.bn_final,
                                        dropout_prob=args.dropout_prob)
    # Create learner
    learner = Learner(data, model,
                      wd=args.weight_decay,
                      loss_func=CustomBCELogitsLoss() if args.binary else nn.CrossEntropyLoss(),
                      metrics=metric)

    # Form layer group for optimization
    meta = model_meta.get(args.model, _default_meta)
    learner.split(meta['split'])
    # Freeze model's head
    if args.pretrained:
        learner.freeze()

    if args.resume:
        learner.load(args.resume)
    if args.unfreeze:
        learner.unfreeze()

    learner.fit_one_cycle(args.epochs, max_lr=slice(None, args.lr, None),
                          div_factor=args.div_factor, final_div=args.final_div_factor)

    learner.save(args.checkpoint)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyroNear Classification Training with Fastai',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input / Output
    parser.add_argument('--data-path', default='./data', help='dataset root folder')
    parser.add_argument('--checkpoint', default='checkpoint', type=str, help='name of output file')
    parser.add_argument('--resume', default=None, help='checkpoint name to resume from')
    # Architecture
    parser.add_argument('--model', default='resnet18', type=str, help='model architecture')
    parser.add_argument("--concat-pool", dest="concat_pool",
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
    # Device
    parser.add_argument('--device', default=None, help='device')
    parser.add_argument("--deterministic", dest="deterministic",
                        help="should the training be performed in deterministic mode",
                        action="store_true")
    # Loader
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('-s', '--resize', default=224, type=int, help='image size after resizing')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers')
    # Optimizer
    parser.add_argument('--lr', default=3e-3, type=float, help='maximum learning rate')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
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
