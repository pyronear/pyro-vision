import unittest

from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from pyronear.datasets.wildfire.wildfireSubSampler import wildfireSubSampler, WildFireSplitter


class WildFireDatasetTester(unittest.TestCase):

    def setUp(self):
        self.path_to_frames = Path(__file__).parent / 'fixtures/'
        self.wildfire_path = Path(__file__).parent / 'fixtures/SubSamplerDS.csv'
        self.wildfire_df = pd.read_csv(self.wildfire_path)

    def test_wildfireSS_correctly_init_from_path(self):
        wildfireSS = wildfireSubSampler(metadata=self.wildfire_path, path_to_frames=self.path_to_frames,
                                        frame_per_seq=2, target_names=['fire'])

        self.assertEqual(len(wildfireSS), 0)
        wildfireSS.computeSubSet()
        self.assertEqual(len(wildfireSS), 800)

    def test_wildfireSS_subSet_ratio(self):
        wildfireSS = wildfireSubSampler(metadata=self.wildfire_path, path_to_frames=self.path_to_frames,
                                        frame_per_seq=2, target_names=['fire'])

        self.assertEqual(len(wildfireSS), 0)
        wildfireSS.computeSubSet(0.8)
        l1 = len(wildfireSS)
        wildfireSS.computeSubSet(0.2)
        l2 = len(wildfireSS)
        self.assertGreater(l1, 800)
        self.assertGreater(l2, l1)
        self.assertGreater(1042, l2)
        wildfireSS.computeSubSet(0)
        self.assertEqual(len(wildfireSS), 1042)

    def test_wildfireSS_correctly_init_from_dataframe(self):
        wildfireSS = wildfireSubSampler(metadata=self.wildfire_path, path_to_frames=self.path_to_frames,
                                        frame_per_seq=2, target_names=['fire'])

        wildfireSS.computeSubSet(0)
        wildfireSS.SubSetImgs[3] = "wildfire_example.jpg"
        wildfireSS.metadata.loc[0, 'imgFile'] = "wildfire_example.jpg"
        # try to get one image of wildfire (item 3 is authorized image fixture)
        observation_3, metadata_3 = wildfireSS[3]
        self.assertIsInstance(observation_3, PIL.Image.Image)  # image correctly loaded ?
        self.assertEqual(observation_3.size, (910, 683))

    def test_wildfireSS_correctly_init_with_transform(self):
        wildfireSS = wildfireSubSampler(metadata=self.wildfire_path, path_to_frames=self.path_to_frames,
                                        frame_per_seq=2, target_names=['fire'],
                                        transform=transforms.Compose([transforms.Resize((100, 66)),
                                                                     transforms.ToTensor()]))
        wildfireSS.computeSubSet(0)
        wildfireSS.SubSetImgs[3] = "wildfire_example.jpg"
        wildfireSS.metadata.loc[0, 'imgFile'] = "wildfire_example.jpg"
        # try to get one image of wildfire (item 3 is authorized image fixture)
        observation_3, metadata_3 = wildfireSS[3]
        self.assertEqual(observation_3.size(), torch.Size((3, 100, 66)))

    def test_invalid_csv_path_raises_exception(self):
        with self.assertRaises(ValueError):
            wildfireSubSampler(metadata='bad_path.csv', path_to_frames=self.path_to_frames,
                               frame_per_seq=2, target_names=['fire'])

    def test_dataloader_can_be_init_with_wildfireSubSampler(self):
        wildfireSS = wildfireSubSampler(metadata=self.wildfire_path, path_to_frames=self.path_to_frames,
                                        frame_per_seq=2, target_names=['fire'])
        DataLoader(wildfireSS, batch_size=64)


class WildFireDatasetSplitter(unittest.TestCase):

    def setUp(self):
        self.path_to_frames = Path(__file__).parent / 'fixtures/'
        self.wildfire_path = Path(__file__).parent / 'fixtures/SubSamplerDS.csv'
        #self.wildfire_df = pd.read_csv(self.wildfire_path)

        self.wildfire = wildfireSubSampler(metadata=self.wildfire_path, path_to_frames=self.path_to_frames,
                                           frame_per_seq=2, target_names=['fire'])
        self.wildfire.computeSubSet()

    def test_consistent_ratios_good_init(self):
        ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        splitter = WildFireSplitter(ratios)
        self.assertEqual(ratios, splitter.ratios)

    def test_inconsistent_ratios_raise_exception(self):
        ratios = {'train': 0.9, 'val': 0.2, 'test': 0.1}  # sum > 1
        with self.assertRaises(ValueError):
            WildFireSplitter(ratios)

    def test_splitting_gives_good_splits_size(self):
        n_samples_expected = {'train': 562, 'val': 120, 'test': 118}
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
