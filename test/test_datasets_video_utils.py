# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import glob
import tempfile
import unittest
import urllib
from collections import namedtuple, Counter
from unittest.mock import patch
from pathlib import Path
import pafy
import pandas as pd
import yaml
import random

from pyrovision.datasets import video_utils


def generate_states_fixture():
    df = pd.DataFrame(columns=['fname', 'fps', 'exploitable', 'fire', 'sequence', 'clf_confidence',
                               'loc_confidence', 'x', 'y', 't', 'stateStart', 'stateEnd', 'fBase'])

    start = 100
    end = 109
    for i in range(14):
        x = random.uniform(200, 500)
        y = random.uniform(200, 500)
        t = random.uniform(0, 100)
        start += 9
        end += 9
        if i < 2:
            b = 6
        else:
            b = 952
        base = str(b) + '.mp4'
        fname = str(b) + '_seq' + str(start - 100) + '_' + str(end + 100) + '.mp4'
        df = df.append({'fname': fname, 'fps': 25, 'exploitable': True, 'fire': 1., 'sequence': 0, 'clf_confidence': 0,
                        'loc_confidence': 0, 'x': x, 'y': y, 't': t, 'stateStart': start,
                        'stateEnd': end, 'fBase': base}, ignore_index=True)

        df.to_csv('test/wildfire_states.csv')


class WildFireFrameExtractorTester(unittest.TestCase):

    @staticmethod
    def download_video_fixtures(path_to_videos):
        video_urls_yaml_url = "https://gist.githubusercontent.com/x0s/2015a7e58d8d3f885b6528d33cd10b2d/raw/"

        with urllib.request.urlopen(video_urls_yaml_url) as video_urls_yaml:
            urls = yaml.safe_load(video_urls_yaml)
        for dest, url in urls.items():
            vid = pafy.new(url)
            stream = vid.getbest()
            print(f'Downloading {stream.get_filesize()/1e6:.2f} MB')
            stream.download((path_to_videos / dest).as_posix())

    @classmethod
    def setUpClass(self):
        self.path_to_videos = Path(__file__).parent / 'videos'
        self.path_to_videos.mkdir(exist_ok=True)

        self.download_video_fixtures(self.path_to_videos)
        generate_states_fixture()
        self.path_to_states = Path(__file__).parent / 'wildfire_states.csv'
        self.path_to_states_count = 14

    def test_pick_frames_randomly(self):
        frame_min, frame_max, f_base = 100, 106, '952.mp4'
        state = pd.Series([frame_min, frame_max, f_base], index=['stateStart', 'stateEnd', 'fBase'])

        for n_frames in [2, 3, 4]:
            # Let's generate frames indexes
            frame_indexes = video_utils.FrameExtractor._pick_frames(state, n_frames, True, allow_duplicates=False)

            # Assert frames indexes are unique
            self.assertEqual(n_frames, frame_indexes.nunique())

            # Assert frames indexes are within allowed range
            self.assertGreaterEqual(frame_indexes.min(), frame_min)
            self.assertLessEqual(frame_indexes.max(), frame_max)

    def test_pick_frames_evenly(self):
        frame_min, frame_max, f_base = 100, 106, '952.mp4'
        state = pd.Series([frame_min, frame_max, f_base], index=['stateStart', 'stateEnd', 'fBase'])
        frame_indexes_expected = {2: [100, 106],
                                  3: [100, 103, 106],
                                  4: [100, 102, 104, 106]}

        for n_frames in [2, 3, 4]:
            # Let's generate frames indexes
            frame_indexes = video_utils.FrameExtractor._pick_frames(state, n_frames, False, allow_duplicates=False)

            # Assert frames indexes are unique
            self.assertEqual(n_frames, frame_indexes.nunique())

            # Assert frames indexes are evenly spaced as expected
            self.assertListEqual(frame_indexes.tolist(), frame_indexes_expected[n_frames])

    def test_pick_too_many_frames_raise_exception(self):
        frame_min, frame_max, f_base = 100, 106, '952.mp4'
        state = pd.Series([frame_min, frame_max, f_base], index=['stateStart', 'stateEnd', 'fBase'])
        n_frames = 8  # Only 7 available: 106-100+1=7

        # For every strategy
        for is_random in [True, False]:
            # Let's try to generate more frames indexes than available
            with self.assertRaises(ValueError):
                video_utils.FrameExtractor._pick_frames(state, n_frames, is_random, allow_duplicates=False)

    def test_pick_too_many_frames_allowed_raise_warning(self):
        frame_min, frame_max, f_base = 100, 106, '952.mp4'
        state = pd.Series([frame_min, frame_max, f_base], index=['stateStart', 'stateEnd', 'fBase'])
        n_frames = 8  # Only 7 available: 106-100+1=7

        # For every strategy
        for is_random in [True, False]:
            # Let's try to generate more frames indexes than available
            with self.assertWarns(Warning):
                video_utils.FrameExtractor._pick_frames(state, n_frames, is_random, allow_duplicates=True)

    def test_frame_extraction_random(self):
        """Extracting frames should produce expected count of images and length of metadata(labels)"""
        for n_frames in [2, 3, 4]:
            frames_count_expected = self.path_to_states_count * n_frames

            frame_extractor = video_utils.FrameExtractor(
                self.path_to_videos,
                self.path_to_states,
                strategy='random',
                n_frames=n_frames
            )

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
        too_many_n_frames = 20

        for strategy in video_utils.FrameExtractor.strategies_allowed:
            frame_extractor = video_utils.FrameExtractor(
                self.path_to_videos,
                self.path_to_states,
                strategy=strategy,
                n_frames=too_many_n_frames
            )

            with tempfile.TemporaryDirectory() as path_to_frames:
                with self.assertRaises(ValueError):
                    (frame_extractor.run(path_to_frames=path_to_frames)
                                    .get_frame_labels())

    def test_frame_extractor_bad_strategy_raise_exception(self):
        """Trying to extract with unknown strategy should raise Exception"""
        with self.assertRaises(ValueError):
            video_utils.FrameExtractor(
                self.path_to_videos,
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

        with patch('pyrovision.datasets.video_utils.cv2.VideoCapture', return_value=VideoCaptureMock):
            with self.assertRaises(IOError):
                # Let's try to extract frames from unreadable video
                frame_extractor = video_utils.FrameExtractor(
                    self.path_to_videos,
                    self.path_to_states,
                    strategy='random',
                    n_frames=2
                )

                with tempfile.TemporaryDirectory() as path_to_frames:
                    (frame_extractor.run(path_to_frames=path_to_frames)
                                    .get_frame_labels())


class FireLabelerTester(unittest.TestCase):

    @staticmethod
    def get_unique_only_count(list_):
        """return count of unique-only elements
        [0, 9, 9, 9, 2, 3, 5, 5] ---> 3
        """
        return len([element for (element, count) in Counter(list_).items() if count == 1])

    def setUp(self):

        # Let's define an immutable structure to set up the fixtures
        WildFireFixture = namedtuple('WildFireFixture', 'descriptions fire_ids_truth')

        # Now let's write the fixtures
        wild_voltaire = WildFireFixture(descriptions="""Small Fire confirmed east of NOAA fire camera at 6:31 PM
Voltaire Fire 7pm to midnight, June 12th 2018
4th hour of the Voltaire Fire as it approaches the urban-wildland interface  at 10 PM
3rd hour of the Voltaire Fire during a growth phase as seen from McClellan at 9 PM
2nd hour of the Voltaire Fire near Carson City from McClellan Peak at  8 PM
Fire near Voltaire Canyon  west of Carson City is confirmed on McClellan Peak camera at  7:43 PM
Leeks Springs camera spins toward early season Rx fire at 11:38 AM""".split('\n'),
                                        fire_ids_truth=[1, 0, 0, 0, 0, 0, 2])

        wild_glen = WildFireFixture(descriptions="""Smoke from Ralph Incident Fire seen after midnight from Angels Roost  4K camera
Sierra at Tahoe fire camera points to the Ralph Incident Fire at 5:40 AM
2nd fire camera points at the Maggie Fire from Midas Peak after 1 PM
Maggie Fire caught shortly after Noon from Jacks Peak fire camera
3rd hour of Glen Fire
2nd hour of Glen Fire
6 hour time lapse of "Glen" fire from fire camera located at Sierra at Tahoe
Start of Glen Fire off of Pioneer Trail seen from Sierra at Tahoe fire camera at 1AM
NOAA fire camera captures smoke plume associated with crash of Beechcraft airplane
Small fire is seen near  South Tahoe High School at 2:51 PM
Flames from the Triple Fire are seen moving closer to ranch as recorded from Jacks 3 PM
Triple Fire spotted from the Jacks Peak fire camera at 2 PM
Jacks Peak's fire camera points to River Ranch Fire towards the SE at 2 PM""".split('\n'),
                                    fire_ids_truth=[0, 0, 1, 1, 2, 2, 2, 2, 4, 5, 3, 3, 6])

        wild_king_fire = WildFireFixture(descriptions="""Zoom to Controlled Burn, Rubicon Oct. 17th, 2014
King Fire, nighttime from Snow Valley Peak, 8 PM Sept. 17th, 2014
King Fire, nighttime from CTC North Tahoe, 8 PM Sept. 17th, 2014
King Fire at sunset from Angel's Roost–Heavenly, 7 PM Sept. 17th, 2014
King Fire at sunset from CTC North Tahoe, 7 PM Sept. 17th, 2014
King Fire at sunset from Snow Valley Peak, 7 PM Sept. 17th, 2014
Cascade Fire from Heavenly 2014 09 24-25
Cascade Fire from SnowValley 2014 09 24-25
Rolling blankets of cloud over Tahoe.
A Heavenly View from Angel's Roost.
Snow Lake smoke as seen from Heavenly Fire Camera
Here comes the rain ...
KingFire view from SnowValley,  The promise of rain ....
Heavenly 20140924 1600-1900 Smoke from Snow Lake Fire near Cascade Lake seen near sunset
Heavenly 20140923 1300 1500
Dense smoke from King Fire chokes NW Tahoe.
KingFire Saturday 09/20, view from East Lake Tahoe.
Shifting high winds bring the King Fire back to life with smoke rolling back into South Lake Tahoe.
Smoke from the King Fire engulfs Tahoe South Shore
KingFire SnowValley 20140917 14:00-22:00
KingFire CTC 20140917 13:00-22:00
KingFire Heavenly 20140917 12:00-20:00
Near-Infrared Night Video of KingFire SnowValley 20140916 23:00-04:00
KingFire Heavenly 20140916 15:00-19:00
KingFire SnowValley 20140916 15:00-18:00
KingFire Heavenly 20140916 11:40-15:00
KingFire CTC 20140916 11:40-15:00
KingFire SnowValley 20140916 12:30-15:00
King Fire Heavenly 20140915 14:32-15:00
King Fire Heavenly 20140915
Bison Fire 2013-07-09 x25 Time Lapse
Bison Fire 2013-07-08 x25 Time Lapse
Bison Fire 2013-07-07 x25 Time Lapse
Bison Fire 2013-07-06 x25 Time Lapse
Bison Fire 2013-07-05 x25 Time Lapse""".split('\n'),
                                         fire_ids_truth=[3, 0, 0, 0, 0, 0, 1, 1, 4, 1,
                                                         1, 5, 0, 1, 1, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         2, 2, 2, 2, 2])

        self.fixtures = [wild_voltaire, wild_glen]
        self.fixtures_long = [wild_king_fire]

    def test_label_correctly_short_dependency(self):
        for (descriptions, fire_ids_truth) in self.fixtures:
            df = pd.DataFrame({'description': descriptions})

            fire_labeler = video_utils.FireLabeler(df, window_size=3)
            fire_labeler.run()
            df = fire_labeler.get_dataframe(column_name='fire_id')
            self.assertListEqual(df['fire_id'].tolist(), fire_ids_truth)
            self.assertEqual(fire_labeler._n_singletons, self.get_unique_only_count(fire_ids_truth))

    def test_label_correctly_long_dependency(self):
        """Test if descriptions are correctly gathered if they are far from each other.
        While keeping in mind videos are ordered in time"""
        for (descriptions, fire_ids_truth) in self.fixtures_long:
            df = pd.DataFrame({'description': descriptions})

            fire_labeler = video_utils.FireLabeler(df, window_size=30)
            fire_labeler.run()
            df = fire_labeler.get_dataframe(column_name='fire_id')
            self.assertListEqual(df['fire_id'].tolist(), fire_ids_truth)
            self.assertEqual(fire_labeler._n_singletons, self.get_unique_only_count(fire_ids_truth))

    def test_firenames_matching(self):
        # It should be robust to space before Fire 'King Fire' & 'KingFire':
        s1 = "King Fire Heavenly 20140915 14:32-15:00"
        s2 = "KingFire SnowValley 20140916 12:30-15:00"
        self.assertTrue(video_utils.FireLabeler.fire_are_matching(s1, s2))

        # If Fire name is hidden in the sentence, it should be a match
        s1 = "Smoke from the King Fire engulfs Tahoe South Shore"
        s2 = "KingFire SnowValley 20140916 12:30-15:00"
        self.assertTrue(video_utils.FireLabeler.fire_are_matching(s1, s2))

        # if fire name is found without being suffixed by Fire, it should match
        s1 = "2nd hour of Glen Fire"
        s2 = '6 hour time lapse of "Glen" fire from fire camera located at Sierra at Tahoe'
        self.assertTrue(video_utils.FireLabeler.fire_are_matching(s1, s2))


if __name__ == '__main__':
    unittest.main()
