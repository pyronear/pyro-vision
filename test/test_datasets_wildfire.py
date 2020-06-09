# -*- coding: utf-8 -*-

# Copyright (c) The pyronear developers.
# This file is dual licensed under the terms of the CeCILL-2.1 and GPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import unittest

from collections import (namedtuple,
                         Counter)

import pandas as pd

from pyronear.datasets.wildfire import FireLabeler


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

            fire_labeler = FireLabeler(df, window_size=3)
            fire_labeler.run()
            df = fire_labeler.get_dataframe(column_name='fire_id')
            self.assertListEqual(df['fire_id'].tolist(), fire_ids_truth)
            self.assertEqual(fire_labeler._n_singletons, self.get_unique_only_count(fire_ids_truth))

    def test_label_correctly_long_dependency(self):
        """Test if descriptions are correctly gathered if they are far from each other.
        While keeping in mind videos are ordered in time"""
        for (descriptions, fire_ids_truth) in self.fixtures_long:
            df = pd.DataFrame({'description': descriptions})

            fire_labeler = FireLabeler(df, window_size=30)
            fire_labeler.run()
            df = fire_labeler.get_dataframe(column_name='fire_id')
            self.assertListEqual(df['fire_id'].tolist(), fire_ids_truth)
            self.assertEqual(fire_labeler._n_singletons, self.get_unique_only_count(fire_ids_truth))

    def test_firenames_matching(self):
        # It should be robust to space before Fire 'King Fire' & 'KingFire':
        s1 = "King Fire Heavenly 20140915 14:32-15:00"
        s2 = "KingFire SnowValley 20140916 12:30-15:00"
        self.assertTrue(FireLabeler.fire_are_matching(s1, s2))

        # If Fire name is hidden in the sentence, it should be a match
        s1 = "Smoke from the King Fire engulfs Tahoe South Shore"
        s2 = "KingFire SnowValley 20140916 12:30-15:00"
        self.assertTrue(FireLabeler.fire_are_matching(s1, s2))

        # if fire name is found without being suffixed by Fire, it should match
        s1 = "2nd hour of Glen Fire"
        s2 = '6 hour time lapse of "Glen" fire from fire camera located at Sierra at Tahoe'
        self.assertTrue(FireLabeler.fire_are_matching(s1, s2))


if __name__ == '__main__':
    unittest.main(FireLabelerTester())
