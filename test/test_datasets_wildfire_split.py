import unittest

from pathlib import Path

import pandas as pd

from pyronear.datasets.wildfire import (WildFireDataset,
                                        WildFireSplitter)


class WildFireDatasetTester(unittest.TestCase):

    def setUp(self):

        self.path_to_frames = 'path/to/frames'
        self.wildfire_path = Path(__file__).parent / 'fixtures/wildfire_dataset.csv'
        self.wildfire_df = pd.read_csv(self.wildfire_path)

    def test_wildfire_correctly_init_from_path(self):
        wildfire = WildFireDataset(metadata=self.wildfire_path,
                                   path_to_frames=self.path_to_frames)

        assert len(wildfire) == 974

    def test_wildfire_correctly_init_from_dataframe(self):
        wildfire = WildFireDataset(metadata=self.wildfire_df,
                                   path_to_frames=self.path_to_frames)

        assert len(wildfire) == 974


if __name__ == '__main__':
    unittest.main()
