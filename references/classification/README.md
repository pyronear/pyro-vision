# Image classification

The goal here is to propose a training script and a dataset to train a wildfire classification model. 

## Setup

Python 3.6 (or higher), [pip](https://pip.pypa.io/en/stable/) and [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) are required to train your models with PyroVision. Install the training-specific dependencies:

```bash
git clone https://github.com/pyronear/pyro-vision.git
pip install -e "pyro-vision/.[training]"
```

## Quick Tour

The script comes with multiples arguments that you can explore:

```bash
python references/classification/train.py --help
```

### OpenFire

You can also use freely our open-source dataset:

```bash
python references/classification/train.py path/to/dataset/folder --openfire --arch rexnet1_0x --lr 1e-3 -b 32 --grad-acc 2 --epochs 100 --device 0 --prefetch-size 512
```

If you prefer to run this in Google Colab, we have a starter notebook for you!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyronear/notebooks/blob/main/pyro-vision/classification_training.ipynb)


### Custom datasets

When we use datasets where we are not owners of the data, we unfortunately cannot share them publicly. However, we still want you to be able to train on your own dataset. If you intend to do so, your dataset should follow the folder's hierarchy below:

```
CustomDataset
├── train
│   └── images
│       ├── 0
│       │   ├── no_fire_train_image_first.jpg
│       │   ├── ...
│       │   └── no_fire_train_image_last.jpg
│       └── 1
│           ├── fire_train_image_first.jpg
│           ├── ...
│           └── fire_train_image_last.jpg
└── val
    └── images
        ├── 0
        │   ├── no_fire_val_image_first.jpg
        │   ├── ...
        │   └── no_fire_val_image_last.jpg
        └── 1
            ├── fire_val_image_first.jpg
            ├── ...
            └── fire_val_image_last.jpg

```

Once this is the case, you can train your model in a similar fashion as for OpenFire:

```bash
python train.py path/to/dataset/folder --model rexnet1_0x --lr 1e-3 -b 16 --epochs 20 --device 0
```

## Available architectures

The list of supported architectures is available [here](https://pyronear.org/pyro-vision/models.html).
