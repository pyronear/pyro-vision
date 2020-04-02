import glob
import tempfile
import unittest

from unittest.mock import patch

from pathlib import Path

import pandas as pd

from pyronear.datasets.wildfire import FrameExtractor


# TODO: test when only two frames available and when n_frames > count of available frames


class WildFireFrameExtractorTester(unittest.TestCase):

    def setUp(self):
        self.path_to_videos = Path(__file__).parent / 'fixtures/videos'  # TODO: download automatically

        self.path_to_states = Path(__file__).parent / 'fixtures/wildfire_states.csv'
        self.path_to_states_count = 14

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

    def test_frame_extraction_random(self):
        """Extracting frames should produce expected count of images and length of metadata(labels)"""
        for n_frames in [2, 3, 4]:
            frames_count_expected = self.path_to_states_count * n_frames

            frame_extractor = FrameExtractor(self.path_to_videos,
                                             self.path_to_states,
                                             strategy='random',
                                             n_frames=n_frames)

            # assert count of frames png files equals to frames registered in labels.csv
            with tempfile.TemporaryDirectory() as path_to_frames:
                labels = (frame_extractor.run(path_to_frames=path_to_frames, seed=69)
                                         .get_frame_labels())

                # Check that count fo frames created equals expected AND frame labels
                frames_count = len(glob.glob1(path_to_frames, "*.png"))
                labels_count = len(labels)
                self.assertEqual(frames_count, labels_count)
                self.assertEqual(frames_count, frames_count_expected)

    def test_frame_extraction_all_strategies_too_many_frames(self):
        """Trying to extract more frames than available should raise Exception"""
        too_many_n_frames = 10

        for strategy in FrameExtractor.strategies_allowed:
            frame_extractor = FrameExtractor(self.path_to_videos,
                                             self.path_to_states,
                                             strategy=strategy,
                                             n_frames=too_many_n_frames)

            with tempfile.TemporaryDirectory() as path_to_frames:
                with self.assertRaises(ValueError):
                    (frame_extractor.run(path_to_frames=path_to_frames)
                                    .get_frame_labels())

    def test_frame_extractor_bad_strategy_raise_exception(self):
        """Trying to extract with unknown strategy should raise Exception"""
        with self.assertRaises(ValueError):
            FrameExtractor(self.path_to_videos,
                           self.path_to_states,
                           strategy='unavailable',
                           n_frames=2)

    def test_frame_video_cannot_be_read_raise_exception(self):
        """Error in reading video frame should raise Exception"""

        class VideoCaptureMock:
            def set(*args):
                pass

            def read():
                return (False, None)

        with patch('pyronear.datasets.wildfire.frame_extractor.cv2.VideoCapture', return_value=VideoCaptureMock):
            with self.assertRaises(IOError):
                # Let's try to extract frames from unreadable video
                frame_extractor = FrameExtractor(self.path_to_videos,
                                                 self.path_to_states,
                                                 strategy='random',
                                                 n_frames=2)

                with tempfile.TemporaryDirectory() as path_to_frames:
                    (frame_extractor.run(path_to_frames=path_to_frames)
                                    .get_frame_labels())


if __name__ == '__main__':
    unittest.main()
