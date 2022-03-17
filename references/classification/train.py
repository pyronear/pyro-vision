# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import os
import time
import datetime

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import holocron
from holocron.optim.wrapper import Lookahead
from holocron.trainer import BinaryClassificationTrainer


def target_transform(target):

    target = torch.tensor(target, dtype=torch.float32)

    return target.unsqueeze(dim=0)


def load_data(data_path, img_size=224, crop_pct=0.8):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=img_size, scale=(crop_pct, 1.0)),
                                           transforms.RandomRotation(degrees=5),
                                           transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize
                                           ])

    val_transforms = transforms.Compose([transforms.Resize(size=(img_size)),
                                         transforms.CenterCrop(size=img_size),
                                         transforms.ToTensor(),
                                         normalize
                                         ])

    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    train_set = ImageFolder(train_dir, train_transforms, target_transform=target_transform)
    val_set = ImageFolder(val_dir, val_transforms, target_transform=target_transform)

    return train_set, val_set


def main(args):

    print(args)

    train_set, val_set = load_data(args.data_path, img_size=args.img_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, drop_last=True,
                                               sampler=RandomSampler(train_set), num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, drop_last=False,
                                             sampler=SequentialSampler(val_set), num_workers=args.workers,
                                             pin_memory=True)

    print("Creating model")
    model = holocron.models.__dict__[args.model](args.pretrained, num_classes=1)

    criterion = nn.BCEWithLogitsLoss()

    # Create the contiguous parameters.
    model_params = [p for p in model.parameters() if p.requires_grad]
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model_params, args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model_params, args.lr,
                                     betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)
    elif args.opt == 'radam':
        optimizer = holocron.optim.RAdam(model_params, args.lr,
                                         betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)
    elif args.opt == 'ranger':
        optimizer = Lookahead(holocron.optim.RAdam(model_params, args.lr,
                                                   betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay))
    elif args.opt == 'tadam':
        optimizer = holocron.optim.TAdam(model_params, args.lr,
                                         betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)

    trainer = BinaryClassificationTrainer(model, train_loader, val_loader, criterion, optimizer,
                                          args.device, args.output_file)
    if args.resume:
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        trainer.load(checkpoint)

    if args.test_only:
        print("Running evaluation")
        eval_metrics = trainer.evaluate()
        print(f"Validation loss: {eval_metrics['val_loss']:.4} "
              f"(Acc@1: {eval_metrics['acc1']:.2%}, Acc@5: {eval_metrics['acc5']:.2%})")
        return

    if args.lr_finder:
        print("Looking for optimal LR")
        trainer.lr_find(args.freeze_until)
        trainer.plot_recorder()
        return

    print("Start training")
    start_time = time.time()
    trainer.fit_n_epochs(args.epochs, args.lr, args.freeze_until)
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Pyronear Classification Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to dataset folder')
    parser.add_argument('--model', default='darknet19', type=str, help='model')
    parser.add_argument('--freeze-until', default=None, type=str, help='Last layer to freeze')
    parser.add_argument('--device', default=None, type=int, help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--img-size', default=224, type=int, help='image size')
    parser.add_argument('--loss', default='crossentropy', type=str, help='loss')
    parser.add_argument('--opt', default='adam', type=str, help='optimizer')
    parser.add_argument('--sched', default='onecycle', type=str, help='Scheduler to be used')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay', dest='weight_decay')
    parser.add_argument("--lr-finder", dest='lr_finder', action='store_true', help="Should you run LR Finder")
    parser.add_argument('--output-file', default='./model.pth', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
                        action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
