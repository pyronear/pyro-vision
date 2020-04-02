import glob
import tempfile
import unittest

from pathlib import Path

import pandas as pd

from pyronear.datasets.wildfire import FrameExtractor


# TODO: test when only two frames available and when n_frames > count of available frames


class WildFireFrameExtractorTester(unittest.TestCase):

    def test_pick_frames_randomly(self):
        frame_min, frame_max = 100, 106
        state = pd.Series([frame_min, frame_max], index=['stateStart', 'stateEnd'])

        for n_frames in [2, 3, 4]:
            # Let's generate frames indexes
            frame_indexes = FrameExtractor._pick_frames(state, n_frames=n_frames, random=True)

            # Assert frames indexes are unique
            self.assertEqual(n_frames, frame_indexes.nunique())

            # Assert frames indexes are within allowed range
            self.assertGreaterEqual(frame_indexes.min(), frame_min)
            self.assertLessEqual(frame_indexes.max(), frame_max)

    def test_pick_frames_evenly(self):
        frame_min, frame_max = 100, 106
        state = pd.Series([frame_min, frame_max], index=['stateStart', 'stateEnd'])
        frame_indexes_expected = {2: [100, 106],
                                  3: [100, 103, 106],
                                  4: [100, 102, 104, 106]}

        for n_frames in [2, 3, 4]:
            # Let's generate frames indexes
            frame_indexes = FrameExtractor._pick_frames(state, n_frames=n_frames, random=False)

            # Assert frames indexes are unique
            self.assertEqual(n_frames, frame_indexes.nunique())

            # Assert frames indexes are evenly spaced as expected
            self.assertListEqual(frame_indexes.tolist(), frame_indexes_expected[n_frames])

    def test_pick_too_many_frames_raise_exception(self):
        frame_min, frame_max = 100, 106
        state = pd.Series([frame_min, frame_max], index=['stateStart', 'stateEnd'])
        n_frames = 8  # Only 7 available: 106-100+1=7

        # For every strategy
        for random in [True, False]:
            # Let's try to generate more frames indexes than available
            with self.assertRaises(ValueError):
                FrameExtractor._pick_frames(state, n_frames=n_frames, random=random)


if __name__ == '__main__':
    unittest.main()
