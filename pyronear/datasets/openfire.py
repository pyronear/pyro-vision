#!usr/bin/python
# -*- coding: utf-8 -*-

import random
from pathlib import Path
import warnings
import json
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
from torchvision.datasets import VisionDataset
from .utils import download_url, download_urls, get_fname

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OpenFire(VisionDataset):
    """Wildfire image Dataset.

    Args:
        root (string): Root directory of dataset where ``OpenFire/processed/training.pt``
            and  ``OpenFire/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        threads (int, optional): If download is set to True, use this amount of threads
            for downloading the dataset.
    """

    url = 'https://gist.githubusercontent.com/frgfm/f53b4f53a1b2dc3bb4f18c006a32ec0d/raw/bf2430074a7de1005534d8cd1fa158e185e43000/openfire_binary.json'
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = [False, True]
    seed = 42

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, threads=16):
        super(OpenFire, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download(threads)

        if not self._check_exists(train):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data = torch.load(self._root.joinpath(self._processed, data_file))

    def __getitem__(self, idx):
        """ Getter function

        Args:
            index (int): Index
        Returns:
            img (torch.Tensor<float>): image tensor
            target (int): dictionary of bboxes and labels' tensors
        """

        # Load image
        img = Image.open(self._root.joinpath(self.data[idx]['path']), mode='r').convert('RGB')
        # Load bboxes & encode label
        target = self.data[idx]['target']
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @property
    def _root(self):
        return Path(self.root)

    @property
    def _raw(self):
        return Path(self.__class__.__name__, 'raw')

    @property
    def _processed(self):
        return Path(self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self, train=True):
        if train:
            return self._root.joinpath(self._processed, self.training_file).is_file()
        else:
            return self._root.joinpath(self._processed, self.test_file).is_file()

    def download(self, threads=None):
        """Download the OpenFire data if it doesn't exist in processed_folder already.

        Args:
            threads (int, optional): Number of threads to use for dataset downloading.
        """

        if self._check_exists(train=True) and self._check_exists(train=False):
            return

        self._root.joinpath(self._raw).mkdir(parents=True, exist_ok=True)
        self._root.joinpath(self._processed).mkdir(parents=True, exist_ok=True)

        # Download annotations
        download_url(self.url, self._root.joinpath(self._raw), filename=self.url.rpartition('/')[-1], verbose=False)
        with open(self._root.joinpath(self._raw, self.url.rpartition('/')[-1]), 'rb') as f:
            annotations = json.load(f)

        # Download actual images
        training_set, test_set = [], []
        img_folder = self._root.joinpath(self._raw, 'images')
        img_folder.mkdir(parents=True, exist_ok=True)
        unavailable_idxs = 0
        # Prepare URL and filenames for multi-processing
        entries = [(a['url'], f"{idx:06}.{get_fname(a['url']).rpartition('.')[-1]}")
                   for idx, a in enumerate(annotations)]
        # Use multiple threads to speed up download
        download_urls(entries, img_folder, threads=threads)
        # Verify downloads
        for idx, annotation in enumerate(annotations):
            img_path = self._raw.joinpath('images', entries[idx][1])
            if self._root.joinpath(img_path).is_file():
                # Encode target
                target = self.class_to_idx[annotation['target']]
                # Aggregate img path and annotations
                data = dict(path=img_path, target=target)
                # Add it to the proper set
                if annotation['is_test']:
                    test_set.append(data)
                else:
                    training_set.append(data)
            else:
                unavailable_idxs += 1
        # HTTP Errors
        if unavailable_idxs > 0:
            warnings.warn((f'{unavailable_idxs}/{len(annotations)} samples could not be downloaded. Please retry later.'))

        # save as torch files
        with open(self._root.joinpath(self._processed, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        # in case test split if not available
        if len(test_set) > 0:
            with open(self._root.joinpath(self._processed, self.test_file), 'wb') as f:
                torch.save(test_set, f)
        else:
            warnings.warn("Unable to find train/test split! All samples were assigned to train set.")

        print('Done!')

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
