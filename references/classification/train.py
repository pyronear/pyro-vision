# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.


import datetime
import logging
import math
import os
import time
from shutil import copyfile

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from codecarbon import track_emissions
from holocron.models.presets import IMAGENET
from holocron.optim import AdamP
from holocron.trainer import BinaryClassificationTrainer
from holocron.transforms.interpolation import RandomZoomOut, Resize, ResizeMethod
from holocron.utils.misc import find_image_size
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from torchvision.transforms.functional import InterpolationMode, resize, to_pil_image

from pyrovision import models
from pyrovision.datasets import OpenFire

logging.getLogger("codecarbon").disabled = True


def target_transform(target):

    target = torch.tensor(target, dtype=torch.float32)

    return target.unsqueeze(dim=0)


def plot_samples(images, targets, num_samples=12):
    # Unnormalize image
    nb_samples = min(num_samples, images.shape[0])
    num_cols = min(nb_samples, 4)
    num_rows = int(math.ceil(nb_samples / num_cols))
    _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
    for idx in range(nb_samples):
        img = images[idx]
        img *= torch.tensor(IMAGENET["std"]).view(-1, 1, 1)
        img += torch.tensor(IMAGENET["mean"]).view(-1, 1, 1)
        img = to_pil_image(img)

        _row = int(idx / num_cols)
        _col = idx - _row * num_cols

        axes[_row][_col].imshow(img)
        axes[_row][_col].axis("off")
        _targets = targets.squeeze()
        if _targets.ndim == 1:
            axes[_row][_col].set_title(_targets[idx].item())
        else:
            class_idcs = torch.where(_targets[idx] > 0)[0]
            _info = [f"{_idx.item()} ({_targets[idx, _idx]:.2f})" for _idx in class_idcs]
            axes[_row][_col].set_title(" ".join(_info))

    plt.show()


@track_emissions()
def main(args):

    print(args)

    torch.backends.cudnn.benchmark = True
    # Data loading
    train_loader, val_loader = None, None

    # Data loading code
    normalize = T.Normalize(mean=IMAGENET["mean"], std=IMAGENET["std"])

    interpolation = InterpolationMode.BILINEAR
    target_size = (args.height, args.width)

    train_transforms = (
        T.Compose(
            [
                # Photometric
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3))], p=0.5),
                # Geometric
                T.RandomHorizontalFlip(),
                RandomZoomOut(target_size, scale=(0.3, 1.0), interpolation=interpolation),
                T.RandomPerspective(distortion_scale=0.2, interpolation=interpolation, p=0.8),
                # Conversion
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float32),
                normalize,
                T.RandomErasing(p=0.9, scale=(0.01, 0.05), value="random"),
            ]
        )
        if not args.find_size
        else None
    )

    val_transforms = T.Compose(
        [
            Resize(target_size, mode=ResizeMethod.PAD, interpolation=interpolation),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32),
            normalize,
        ]
    )

    print("Loading data")
    # Prefetch images and resize them to avoid RAM overload
    prefetch_fn = None
    if isinstance(args.prefetch_size, int):

        def prefetch_fn(img_paths):
            # Unpack paths
            src_path, dest_path = img_paths
            if os.path.exists(dest_path):
                try:
                    img = Image.open(dest_path, mode="r").convert("RGB")
                    if min(img.size) <= args.prefetch_size:
                        return
                except Exception:
                    pass

            img = Image.open(src_path, mode="r").convert("RGB")
            # Resize & save
            if all(dim > args.prefetch_size for dim in img.size):
                resized_img = resize(img, args.prefetch_size, interpolation=interpolation)
                resized_img.save(dest_path)
            # Copy
            else:
                copyfile(src_path, dest_path)

    if not (args.find_size or args.test_only):
        st = time.time()
        if args.openfire:

            train_set = OpenFire(
                args.data_path, train=True, download=True, transform=train_transforms, prefetch_fn=prefetch_fn
            )
        else:
            train_dir = os.path.join(args.data_path, "train")
            train_set = ImageFolder(train_dir, train_transforms, target_transform=target_transform)

        # Suggest size
        if args.find_size:
            print("Looking for optimal image size")
            find_image_size(train_set)
            return

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            drop_last=True,
            sampler=RandomSampler(train_set),
            num_workers=args.workers,
            pin_memory=True,
        )

        print(
            f"Training set loaded in {time.time() - st:.2f}s ({len(train_set)} samples in {len(train_loader)} batches)"
        )

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    if not (args.find_lr):
        st = time.time()
        if args.openfire:
            val_set = OpenFire(
                args.data_path, train=False, download=True, transform=val_transforms, prefetch_fn=prefetch_fn
            )
        else:
            val_dir = os.path.join(args.data_path, "val")
            val_set = ImageFolder(val_dir, val_transforms, target_transform=target_transform)

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            drop_last=False,
            sampler=SequentialSampler(val_set),
            num_workers=args.workers,
            pin_memory=True,
        )
        print(f"Validation set loaded in {time.time() - st:.2f}s ({len(val_set)} samples in {len(val_loader)} batches)")

    print("Creating model")
    model = models.__dict__[args.arch](args.pretrained, num_classes=1)

    criterion = nn.BCEWithLogitsLoss()

    # Create the contiguous parameters.
    model_params = [p for p in model.parameters() if p.requires_grad]
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(model_params, args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt == "radam":
        optimizer = torch.optim.RAdam(
            model_params, args.lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay
        )
    elif args.opt == "adamp":
        optimizer = AdamP(model_params, args.lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)

    log_wb = lambda metrics: wandb.log(metrics) if args.wb else None
    trainer = BinaryClassificationTrainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.device,
        args.output_file,
        gradient_acc=args.grad_acc,
        amp=args.amp,
        on_epoch_end=log_wb,
    )

    if args.resume:
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        trainer.load(checkpoint)

    if args.test_only:
        print("Running evaluation")
        eval_metrics = trainer.evaluate()
        print(trainer._eval_metrics_str(eval_metrics))
        return

    if args.find_lr:
        print("Looking for optimal LR")
        trainer.find_lr(args.freeze_until, num_it=min(len(train_loader), 100), norm_weight_decay=args.norm_wd)
        trainer.plot_recorder()
        return

    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}-{current_time}" if args.name is None else args.name

    # W&B
    if args.wb:

        run = wandb.init(
            name=exp_name,
            project="pyrovision-image-classification",
            config={
                "learning_rate": args.lr,
                "scheduler": args.sched,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.grad_acc,
                "architecture": args.arch,
                "input_size": args.img_size,
                "prefetch_size": args.prefetch_size,
                "optimizer": args.opt,
                "dataset": "openfire" if args.openfire else "custom",
                "loss": "bce",
            },
        )

    print("Start training")
    start_time = time.time()
    trainer.fit_n_epochs(args.epochs, args.lr, args.freeze_until, args.sched, norm_weight_decay=args.norm_wd)
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time_str}")

    if args.wb:
        run.finish()


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pyronear Classification Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("data_path", type=str, help="path to dataset folder")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--arch", default="rexnet1_0x", type=str, help="model")
    parser.add_argument("--openfire", help="whether OpenFire should be used", action="store_true")
    parser.add_argument("--freeze-until", default=None, type=str, help="Last layer to freeze")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("--grad-acc", default=1, type=int, help="Number of batches to accumulate the gradient of")
    parser.add_argument("--epochs", default=20, type=int, help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, help="number of data loading workers")
    parser.add_argument("--height", default=256, type=int, help="image height")
    parser.add_argument("--width", default=384, type=int, help="image width")
    parser.add_argument(
        "--prefetch-size", default=None, type=int, help="prefetched images will be resized to lower RAM usage"
    )
    parser.add_argument("--opt", default="adamp", type=str, help="optimizer")
    parser.add_argument("--sched", default="onecycle", type=str, help="Scheduler to be used")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("--norm-wd", default=None, type=float, help="weight decay of norm parameters")
    parser.add_argument("--find-lr", dest="find_lr", action="store_true", help="Should you run LR Finder")
    parser.add_argument("--find-size", dest="find_size", action="store_true", help="Should you run Image size Finder")
    parser.add_argument("--show-samples", action="store_true", help="Whether training samples should be displayed")
    parser.add_argument("--output-file", default="./checkpoint.pth", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument(
        "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true"
    )
    parser.add_argument("--amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument("--wb", action="store_true", help="Log to Weights & Biases")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
