import numpy as np
from pathlib import Path

import torch
import warnings
from fastai import vision
from pyronear.datasets import OpenFire

np.random.seed(42)
# Disable warnings from fastai using deprecated functions for PyTorch>=1.3
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def main(args):
    path = Path(args.data_path)

    # Aggregate path and labels into list for fastai ImageDataBunch
    fnames, labels = [], []
    for sample in OpenFire(root=args.data_path, train=True, download=True).data:
        fnames.append(path.joinpath(sample['path']))
        labels.append(sample['target'])
    for sample in OpenFire(root=args.data_path, train=False, download=True).data:
        fnames.append(path.joinpath(sample['path']))
        labels.append(sample['target'])

    data = vision.ImageDataBunch.from_lists(args.data_path, fnames=fnames, labels=labels,
                                            valid_pct=0.2, bs=args.batch_size, ds_tfms=vision.get_transforms(),
                                            size=224, num_workers=args.workers).normalize(vision.imagenet_stats)

    # # Train & test sets
    learn = vision.cnn_learner(data, vision.models.__dict__[args.model],
                               pretrained=args.pretrained,
                               wd=args.weight_decay,
                               ps=args.dropout_prob,
                               concat_pool=args.concat_pool,
                               metrics=vision.error_rate)

    learn.fit_one_cycle(args.epochs, max_lr=slice(None, args.lr, None),
                        div_factor=args.div_factor)

    learn.save(args.checkpoint)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyroNear Classification Training with Fastai')
    parser.add_argument('--data-path', default='./data', help='dataset')
    parser.add_argument('--model', default='resnet18', type=str, help='model')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=3e-3, type=float, help='initial learning rate')
    parser.add_argument("--concat-pool", dest="concat_pool",
        help="Use pre-trained models from the modelzoo",
        action="store_true"
    )
    parser.add_argument('--dropout-prob', default=0.5, type=float, help='dropout rate of last FC layer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')
    parser.add_argument('--div-factor', default=25., type=float, help='div factor of OneCycle policy')
    parser.add_argument('--checkpoint', default='checkpoint', type=str, help='name of output file')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)