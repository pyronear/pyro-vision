# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


import warnings
import re
from itertools import combinations
from functools import partial
from pathlib import Path
from typing import ClassVar, List, Union

import cv2
import numpy as np
import pandas as pd

__all__ = ['FrameExtractor', 'FireLabeler']


class FrameExtractor:
    """Extract frames from wildfire videos according to a strategy

    Parameters
    ----------
    path_to_videos: str or Path
        Path leading to the full wildfire videos

    path_to_states: str or Path
        Path leading to CSV containing states.
        A state describes the scene between two frames keeping same labels
        Ex: Between frame 27 and 56, Fire seeable with confidence
        at position (x, y) but not located with confidence.
        Expected columns are:
        - stateStart and stateEnd: lower and upper frame labels encircling the state
        - fBase: name of the full videos from which to extract the frames

    strategy: str
        strategy to use in order to extract the frames.
        For now, two strategies are available:
        - 'random': extract randomly frames per state
        - 'uniform': extract evenly frames per state

    n_frames: int
        Number of frames as a parameter for extraction(for now, this is per state)

    Note: Here is an example of what a states CSV look like:
    states:
        fname               fBase   fps fire  sequence  clf_confidence  loc_confidence  exploitable    x        y       t      stateStart   stateEnd
        0_seq0_344.mp4      0.mp4   25  0     0         1               0               True          609.404   450.282 0.167  4            344
        0_seq1061_1475.mp4  0.mp4   25  1     0         1               0               True          1027.524  558.621 2.015  1111         1449
        0_seq446_810.mp4    0.mp4   25  1     0         1               0               True          695.737   609.404 1.473  483          810


    Example
    -------
    frame_extractor = FrameExtractor("../WildFire",
                                     'jean_lulu_with_seq_01.states.csv',
                                     strategy='random',
                                     n_frames=2)

    labels = (frame_extractor.run(path_to_frames='koukou')
                             .get_frame_labels())
    """ # noqa
    strategies_allowed: ClassVar[List[str]] = ['random', 'evenly']

    def __init__(self,
                 path_to_videos: Union[str, Path],
                 path_to_states: Union[str, Path],
                 strategy: str = 'random',
                 n_frames: int = 2):

        self.path_to_videos = Path(path_to_videos)
        self.path_to_states = Path(path_to_states)
        self.strategy = strategy
        self.n_frames = n_frames

        if self.strategy not in self.strategies_allowed:
            raise ValueError(f"Strategy {self.strategy} is unknown."
                             f"Please choose from : {', '.join(self.strategies_allowed)}")

        self.states = pd.read_csv(path_to_states)

    def run(self, path_to_frames: Union[str, Path], allow_duplicates: bool = False, seed: int = 42):
        """Run the frame extraction on the videos according to given strategy and states

        path_to_frames: str or Path, path where to save the frames

        allow_duplicates: bool (default: False), whether or not to allow frames duplicates
            (One unique image(frame) may match multiple frames registered in labels

        seed: int, seed for random picking (default: 42)
        """
        # Define frames to extract given a strategy
        if (self.strategy == 'random'):
            random = True
        elif(self.strategy == 'evenly'):
            random = False

        labels = self._get_frame_labels(self.states, self.n_frames, random, allow_duplicates, seed)

        # Write labels
        path_to_frames = Path(path_to_frames)
        path_to_frames.mkdir(exist_ok=True)

        basename = self.path_to_states.stem
        path_to_frame_labels = path_to_frames / f'{basename}.labels.csv'
        print(f'Writing frame labels to {path_to_frame_labels}')
        labels.to_csv(path_to_frame_labels, index=False)
        self._labels = labels

        # Write frames
        print(f'Extracting {self.n_frames} frames per state ({len(labels)} in total) to {path_to_frames}')
        self._write_frames(labels, path_to_frames)
        return self

    def get_frame_labels(self) -> pd.DataFrame:
        return self._labels

    @staticmethod
    def _pick_frames(state: pd.Series, n_frames: int, random: bool,
                     allow_duplicates: bool, seed: int = 42) -> pd.Series:
        """
        Return a Series with the list of selected frames for the given state (n_frames x 1)

        Parameters
        ----------
        state: pd.Series containing stateStart, stateEnd and fBase

        n_frames: number of frames to pick

        allow_duplicates: bool, Whether or not to allow frames duplicates
            (One unique image(frame) may match multiple frames registered in labels

        random: bool
            Pick frames randomly or according to np.linspace,
            e.g. first if n_frames = 1, + last if n_frames = 2, + middle if n_frames = 3, etc

        seed: int, seed for random picking (default: 42)
        """
        np.random.seed(seed)

        # Trying to set a valid frame range
        frames_range = range(state.stateStart, state.stateEnd + 1)
        frames_range_len = len(frames_range)
        if frames_range_len < n_frames:
            if not allow_duplicates:
                raise ValueError(f"Not enough frames available({frames_range_len})"
                                 f" in the state to extract {n_frames} frames from {state.fBase}")
            else:
                warnings.warn(f"frames available({frames_range_len}) in the state"
                              f"are lower than the ones to extract ({n_frames}) from {state.fBase}."
                              f"Warning, they will be duplicates registered in labels but"
                              f"no duplicates as images because of unique filenames")

        # Let's pick frames according to strategy
        if random:
            # randomly select unique frame numbers within state range
            return pd.Series(np.random.choice(frames_range, size=n_frames, replace=allow_duplicates))
        else:
            # select evenly spaced frames within state range
            return pd.Series(np.linspace(state.stateStart, state.stateEnd, n_frames, dtype=int))

    def _get_frame_labels(self, states: pd.DataFrame, n_frames: int, random: bool,
                          allow_duplicates: bool = False, seed: int = 42) -> pd.DataFrame:
        """
        Given a DataFrame with states, call _pickFrames to create a DataFrame with
        n_frames per state containing the state information, filename and
        imgFile (the name of the file to be used when writing an image)

        Parameters
        ----------
        states: DataFrame containing fBase, stateStart, stateEnd

        n_frames: int, number of frames per state

        random: bool, pick frames randomly(True) or evenly(False)

        allow_duplicates: bool (default: False), whether or not to allow frames duplicates
            (One unique image(frame) may match multiple frames registered in labels

        seed: int, seed for pseudorandom generator
        """
        pick_frames_for_one_state = partial(self._pick_frames, n_frames=n_frames, random=random,
                                            allow_duplicates=allow_duplicates, seed=seed)
        # DataFrame containing columns (0..n_frames - 1)
        frames = states.apply(pick_frames_for_one_state, axis=1)  # (n_states x n_frames)

        # Merge states and frames and transform each value of the new columns into a row
        # Drop the new column 'variable' that represents the column name in frames
        df = pd.melt(states.join(frames), id_vars=states.columns,
                     value_vars=range(n_frames), value_name='frame').drop(columns=['variable'])

        # Add image file name
        df['imgFile'] = df.apply(lambda x: Path(x.fBase).stem + f'_frame{x.frame}.png', axis=1)
        return df.sort_values(['fBase', 'frame'])

    def _write_frames(self, labels: pd.DataFrame, path_to_frames: Union[str, Path]) -> None:
        """Extract frames from videos and write frames as
        <path_to_frames>/<fBase>_frame<frame>.png

        Parameters
        ----------
        labels: Pandas DataFrame containing:
            - fBase: filename of unsplit video (ex: 3.mp4)
            - frame: indexex of the frames to extract (ex: 56)
            - imgFile: filenames to save the frames (ex: 3_frame56.png)

        path_to_frames: str, output directory. Created if needed
        """
        path_to_frames = Path(path_to_frames)
        path_to_frames.mkdir(exist_ok=True)

        # For each video (ex: 3.mp4)
        for name, group in labels.groupby('fBase'):
            # Get the video
            movie = cv2.VideoCapture((self.path_to_videos / name).as_posix())
            # For each state
            for row in group.itertuples():
                # Position the video at the current frame
                movie.set(cv2.CAP_PROP_POS_FRAMES, row.frame)
                success, frame = movie.read()
                # Save the frame
                if success:
                    cv2.imwrite((path_to_frames / row.imgFile).as_posix(), frame)
                else:
                    raise IOError(f'Could not read frame {row.frame} from {name}')


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
