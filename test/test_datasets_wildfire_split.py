# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import unittest

from pathlib import Path

import pandas as pd
import PIL
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from pyrovision.datasets.wildfire import (WildFireDataset,
                                        WildFireSplitter)


class WildFireDatasetTester(unittest.TestCase):

    def setUp(self):
        self.path_to_frames = Path(__file__).parent / 'fixtures/'
        self.path_to_frames_str = str(self.path_to_frames)
        self.wildfire_path = Path(__file__).parent / 'fixtures/wildfire_dataset.csv'
        self.wildfire_df = pd.read_csv(self.wildfire_path)

    def test_wildfire_correctly_init_from_path(self):
        for path_to_frames in [self.path_to_frames, self.path_to_frames_str]:
            wildfire = WildFireDataset(metadata=self.wildfire_path,
                                       path_to_frames=path_to_frames)

            self.assertEqual(len(wildfire), 974)
            self.assertEqual(len(wildfire[3]), 2)

    def test_wildfire_correctly_init_from_dataframe(self):
        for path_to_frames in [self.path_to_frames, self.path_to_frames_str]:
            wildfire = WildFireDataset(metadata=self.wildfire_df,
                                       path_to_frames=path_to_frames)

            self.assertEqual(len(wildfire), 974)
            self.assertEqual(len(wildfire[3]), 2)

        # try to get one image of wildfire (item 3 is authorized image fixture)
        observation_3, metadata_3 = wildfire[3]
        self.assertIsInstance(observation_3, PIL.Image.Image)  # image correctly loaded ?
        self.assertEqual(observation_3.size, (910, 683))
        self.assertTrue(torch.equal(metadata_3, torch.tensor([0])))  # metadata correctly loaded ?

    def test_wildfire_correctly_init_with_multiple_targets(self):
        wildfire = WildFireDataset(metadata=self.wildfire_df,
                                   path_to_frames=self.path_to_frames,
                                   transform=transforms.ToTensor(),
                                   target_names=['fire', 'fire_id'])

        self.assertEqual(len(wildfire), 974)

        # try to get one image of wildfire (item 3 is authorized image fixture)
        observation_3, metadata_3 = wildfire[3]
        self.assertIsInstance(observation_3, torch.Tensor)  # image correctly loaded ?
        self.assertEqual(observation_3.size(), torch.Size([3, 683, 910]))
        self.assertTrue(torch.equal(metadata_3, torch.tensor([0, 96])))  # metadata correctly loaded ?

    def test_invalid_csv_path_raises_exception(self):
        with self.assertRaises(ValueError):
            WildFireDataset(metadata='bad_path.csv',
                            path_to_frames=self.path_to_frames)

    def test_wildfire_correctly_init_with_transform(self):
        wildfire = WildFireDataset(metadata=self.wildfire_path,
                                   path_to_frames=self.path_to_frames,
                                   transform=transforms.Compose([transforms.Resize((100, 66)),
                                                                 transforms.ToTensor()]))

        observation_3, metadata_3 = wildfire[3]
        self.assertEqual(observation_3.size(), torch.Size((3, 100, 66)))

    def test_dataloader_can_be_init_with_wildfire(self):
        wildfire = WildFireDataset(metadata=self.wildfire_path,
                                   path_to_frames=self.path_to_frames)
        DataLoader(wildfire, batch_size=64)


class WildFireDatasetSplitter(unittest.TestCase):

    def setUp(self):
        self.path_to_frames = Path(__file__).parent / 'fixtures/'
        self.wildfire_path = Path(__file__).parent / 'fixtures/wildfire_dataset.csv'
        #self.wildfire_df = pd.read_csv(self.wildfire_path)

        self.wildfire = WildFireDataset(metadata=self.wildfire_path,
                                        path_to_frames=self.path_to_frames)

    def test_consistent_ratios_good_init(self):
        ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        splitter = WildFireSplitter(ratios)
        self.assertEqual(ratios, splitter.ratios)

    def test_inconsistent_ratios_raise_exception(self):
        ratios = {'train': 0.9, 'val': 0.2, 'test': 0.1}  # sum > 1
        with self.assertRaises(ValueError):
            WildFireSplitter(ratios)

    def test_splitting_with_test_to_zero(self):
        ratios = {'train': 0.81, 'val': 0.19, 'test': 0}

        splitter = WildFireSplitter(ratios, seed=42)
        splitter.fit(self.wildfire)

        for (set_, ratio_) in splitter.ratios_.items():
            self.assertAlmostEqual(ratio_, ratios[set_], places=2)

    def test_splitting_gives_good_splits_size(self):
        n_samples_expected = {'train': 684, 'val': 147, 'test': 143}
        ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

        splitter = WildFireSplitter(ratios, seed=42)
        splitter.fit(self.wildfire)

        self.assertEqual(splitter.n_samples_, n_samples_expected)
        for (set_, ratio_) in splitter.ratios_.items():
            self.assertAlmostEqual(ratio_, ratios[set_], places=2)

    def test_splitting_working_with_transforms(self):
        ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        transforms_expected = {'train': transforms.RandomCrop(10), 'val': None, 'test': None}

        splitter = WildFireSplitter(ratios, transforms=transforms_expected)
        splitter.fit(self.wildfire)

        for (set_, transform_expected) in transforms_expected.items():
            self.assertIs(getattr(splitter, set_).transform, transform_expected)

    def test_splitting_with_unavailable_algorithm_raise_exception(self):
        ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

        splitter = WildFireSplitter(ratios, algorithm='wtf')
        with self.assertRaises(ValueError):
            splitter.fit(self.wildfire)


if __name__ == '__main__':
    unittest.main()
