# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import re
from itertools import combinations

import numpy as np


class FireLabeler:
    """Automatically labelize WildFire dataset based on video descriptions

    It will create a new column containing Fire ids that try to identify the videos
    illustrating same real fire.

    An instance of the Labeler is bound to a dataframe but can be
    run several times, in order to vary the window size for instance.

    Parameters
    ----------
    df: pandas.DataFrame
        This DataFrame should:
            - be indexed from 0 to (max number of videos-1) (range-like)
            - contain the video descriptions in the uploading order.
                The closer the upload, the more likely videos represent same fire
            - have a column named 'description' containing the description of the videos

    window_size: in (default=10)
        Count of video descriptions to use to determine if they represent same fire.

    Attributes
    ----------
    n_videos_total: int
        Count of videos (rows found in dataframe)

    n_windows: int
        Count of windows

    _n_singletons: int >= 0
        Count of videos not grouped with at least one other
        Learnt attribute, available after run()

    Examples
    --------
    df = pd.read_csv("WildFire.csv", index_col=0)
    fire_labeler = FireLabeler(df, window_size=30)
    fire_labeler.run()
    df_on_fire = fire_labeler.get_new_dataframe(column_name='fire_id')

    df_on_fire_short = (fire_labeler.reset(window_size=6)
                                    .run()
                                    .get_dataframe(column_name='fire_id'))
    """
    def __init__(self, df, window_size=10):
        self.df = df
        self.window_size = window_size

        self.reset()

    def reset(self, window_size=None):
        """Reset the labeler
        Reset fire ids previously found(if any) and set again all dimensions
        Especially useful, if we want to try another window size"""
        self.window_size = window_size or self.window_size

        self.n_videos_total = self.df.shape[0]
        self.n_windows = self.n_videos_total - self.window_size + 1  # n_windows + windows_size < n_videos_total

        # Store new column for fire ids starting at 0 (-1 for unassigned)
        self.fire_ids = np.full((self.n_videos_total), -1)
        return self

    def run(self):
        """ Run the labelisation of the fire depending on the descriptions

        For every combination of descriptions(strings) in every sliding window
        guess if they belong to same fire (fire_id)

        Note: algo complexity can be improved (ex: by memoizing sliding combinations).
        But, for now, processing is fast enough(<1min) when used with large window-size<100
        """

        # sliding iterator over the video indexes. Example with window_size=4:
        # [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], ...]
        window_idx_it = (range(start, start + self.window_size) for start in range(self.n_windows))

        current_fire_id = 0  # start grouping feu id at 0
        for window_idx in window_idx_it:  # for every window of videos

            # dict with {id: description(string)}
            id_to_descriptions = dict(zip(window_idx, self.df.loc[window_idx, 'description']))

            # for every possible couple of descriptions in the current window
            for id_s1, id_s2 in combinations(id_to_descriptions, 2):
                fire_match = self.fire_are_matching(id_to_descriptions[id_s1],
                                                    id_to_descriptions[id_s2])
                if fire_match:
                    # if s1 or s2 has already a fire_id, assign it
                    if self.fire_ids[id_s1] != -1:
                        self.fire_ids[id_s2] = self.fire_ids[id_s1]
                    elif self.fire_ids[id_s2] != -1:
                        self.fire_ids[id_s1] = self.fire_ids[id_s2]
                    else:  # else we add new fire_id (first encounter)
                        self.fire_ids[id_s1] = current_fire_id
                        self.fire_ids[id_s2] = current_fire_id
                        current_fire_id = current_fire_id + 1

        # Now labeling the singletons
        self._n_singletons = -1 * self.fire_ids[self.fire_ids == -1].sum()

        length = len(self.fire_ids[self.fire_ids == -1])
        self.fire_ids[self.fire_ids == -1] = range(current_fire_id, current_fire_id + length)
        assert self.fire_ids[self.fire_ids == -1].sum() == 0, "Singletons escaped indexation!"
        return self

    @staticmethod
    def fire_are_matching(s1, s2):
        """Compare two fire videos descriptions and guess if they match"""

        # regexp catching fire name (ex: Goose Fire)
        p = re.compile(r"(?P<firename>\w+\sFire)")  # compile once

        def get_firename(string):
            """try to extract fire name"""
            result = p.search(string)
            if result is not None:
                return result.group('firename')
            return None

        firename_s1 = get_firename(s1)
        firename_s2 = get_firename(s2)

        # Conditions:
        # - We need to find at least one firename to compare descriptions(!=None)
        # - if same fire names found, it's a match.
        # - if 'Glen Fire' is found and 'Glen' is also found, it's a match
        # - if 'King Fire' is found and 'KingFire' as well, it's a match
        firenames_match = ((firename_s1 is not None
                            and ((firename_s1 == firename_s2)
                                 or firename_s1.split(' ')[0] in s2
                                 or firename_s1.replace(' ', '') in s2))
                           or (
                           firename_s2 is not None
                           and ((firename_s1 == firename_s2)
                                or firename_s2.split(' ')[0] in s1
                                or firename_s2.replace(' ', '') in s1)))

        return firenames_match

    def get_dataframe(self, column_name='fire_id'):
        """Return the new dataframe complemented with a column(Series) containing the Fire ids"""
        self.df[column_name] = self.fire_ids
        return self.df
