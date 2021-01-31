# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import unittest
from pathlib import Path
import pandas as pd

from pyrovision.datasets.wildfire import computeSubSet


class WildFireSubSamplerTester(unittest.TestCase):

    def setUp(self):
        self.path_to_frames = Path(__file__).parent / 'fixtures/'
        self.wildfire_path = Path(__file__).parent / 'fixtures/subsampler.csv'
        self.wildfire_df = pd.read_csv(self.wildfire_path)

    def test_good_size_after_subsamping(self):
        self.assertEqual(len(self.wildfire_df), 1999)
        metadataSS = computeSubSet(self.wildfire_df, 2)

        self.assertEqual(len(metadataSS), 400)

    def test_metadata_changes_each_time(self):
        metadataSS_1 = computeSubSet(self.wildfire_path, 2, seed=1)
        metadataSS_2 = computeSubSet(self.wildfire_path, 2, seed=2)

        self.assertEqual(len(metadataSS_1), 400)
        self.assertEqual(len(metadataSS_2), 400)
        self.assertFalse(metadataSS_1['imgFile'].values.tolist() == metadataSS_2['imgFile'].values.tolist())

    def test_metadata_does_not_changes_with_same_seed(self):
        metadataSS_1 = computeSubSet(self.wildfire_path, 2, seed=1)
        metadataSS_2 = computeSubSet(self.wildfire_path, 2, seed=1)

        self.assertEqual(len(metadataSS_1), 400)
        self.assertEqual(len(metadataSS_2), 400)
        self.assertTrue(metadataSS_1['imgFile'].values.tolist() == metadataSS_2['imgFile'].values.tolist())

    def test_increase_not_fire_semples(self):
        metadataSS = computeSubSet(self.wildfire_df, 2, 1)

        self.assertGreater(len(metadataSS), 400)


if __name__ == '__main__':
    unittest.main()
