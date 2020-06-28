# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

from pathlib import Path
import pandas as pd

from pyronear.datasets.wildfire import WildFireDataset, WildFireSubSampler


class WildFireDatasetSubSampler(unittest.TestCase):

    def setUp(self):
        self.path_to_frames = Path(__file__).parent / 'fixtures/'
        self.wildfire_path = Path(__file__).parent / 'fixtures/subsampler.csv'
        self.wildfire_df = pd.read_csv(self.wildfire_path)

    def test_good_size_after_subsamping(self):
        wildfire = WildFireDataset(metadata=self.wildfire_path,
                                   path_to_frames=self.path_to_frames)

        self.assertEqual(len(wildfire), 1999)

        subsampler = WildFireSubSampler(wildfire.metadata, 2)
        wildfire.metadata = subsampler.computeSubSet()

        self.assertEqual(len(wildfire), 400)

    def test_metadata_changes_each_time(self):
        wildfire = WildFireDataset(metadata=self.wildfire_path,
                                   path_to_frames=self.path_to_frames)

        subsampler = WildFireSubSampler(wildfire.metadata, 2)
        wildfire.metadata = subsampler.computeSubSet()

        meta2 = subsampler.computeSubSet()

        self.assertEqual(len(meta2), 400)
        self.assertFalse(wildfire.metadata['imgFile'].values.tolist() == meta2['imgFile'].values.tolist())

    def test_increase_not_fire_semples(self):
        wildfire = WildFireDataset(metadata=self.wildfire_path,
                                   path_to_frames=self.path_to_frames)

        subsampler = WildFireSubSampler(wildfire.metadata, 2, 1)
        wildfire.metadata = subsampler.computeSubSet()

        self.assertGreater(len(wildfire), 400)


if __name__ == '__main__':
    unittest.main()
