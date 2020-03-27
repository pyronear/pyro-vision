import unittest

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from pyronear.datasets.wildfire import (WildFireDataset,
                                        WildFireSplitter)


class WildFireDatasetTester(unittest.TestCase):

    def setUp(self):
        self.path_to_frames = Path(__file__).parent / 'fixtures/'
        self.wildfire_path = Path(__file__).parent / 'fixtures/wildfire_dataset.csv'
        self.wildfire_df = pd.read_csv(self.wildfire_path)

    def test_wildfire_correctly_init_from_path(self):
        wildfire = WildFireDataset(metadata=self.wildfire_path,
                                   path_to_frames=self.path_to_frames)

        self.assertEqual(len(wildfire), 974)

    def test_wildfire_correctly_init_from_dataframe(self):
        wildfire = WildFireDataset(metadata=self.wildfire_df,
                                   path_to_frames=self.path_to_frames)

        self.assertEqual(len(wildfire), 974)

        # try to get one image of wildfire (item 3 is authorized image fixture)
        observation_3, metadata_3 = wildfire[3]
        self.assertIsInstance(observation_3, torch.Tensor)  # image correctly loaded ?
        self.assertEqual(observation_3.size(), torch.Size([3, 683, 910]))
        self.assertTrue(torch.equal(metadata_3, torch.tensor([0])))  # metadata correctly loaded ?

    def test_wildfire_correctly_init_with_multiple_targets(self):
        wildfire = WildFireDataset(metadata=self.wildfire_df,
                                   path_to_frames=self.path_to_frames,
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

    def test_splitting_gives_good_splits_size(self):
        n_samples_expected = {'train': 685, 'val': 147, 'test': 142}
        ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

        splitter = WildFireSplitter(ratios)
        splitter.fit(self.wildfire)

        self.assertEqual(splitter.n_samples_, n_samples_expected)
        for (set_, ratio_) in splitter.ratios_.items():
            self.assertAlmostEqual(ratio_, ratios[set_], places=2)


if __name__ == '__main__':
    unittest.main()
