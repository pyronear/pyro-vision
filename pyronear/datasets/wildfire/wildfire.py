# -*- coding: utf-8 -*-

import warnings

import numpy as np
import pandas as pd
import torch
from random import SystemRandom
import random
from torch.utils.data import Dataset

from ..utils import VisionMixin
from .split_strategy import ExhaustSplitStrategy


class WildFireDataset(Dataset, VisionMixin):
    """WildFire dataset that can be fed to a torch model

    Parameters
    ----------
    metadata: str or Pandas.DataFrame
        Path leading to a CSV that will contain the metada of the dataset or directly the DataFrame.
        Field that should be present:
            'imgFile': path_to_frame
            'fire_id': fire index

    target_names: list,
        List of the columns that can be found in metadata CSV and that represent targets
        we want to return when accessing the datasets
        If left to None, will be set to ['fire']
        Example: ['fire', 'clf_confidence', 'loc_confidence', 'x', 'y']

    path_to_frames: str
        Path leading to the directory containing the frames referenced in metadata 'imgFile':

    transform: object, optional
        Transformations to apply to the frames (ie: torchvision.transforms)
    """
    def __init__(self, metadata, path_to_frames, target_names=None, transform=None):
        if isinstance(metadata, pd.DataFrame):
            self.metadata = metadata
        else:
            try:
                self.metadata = pd.read_csv(metadata)
            except (ValueError, FileNotFoundError):
                raise ValueError(f"Invalid path to CSV containing metadata. Please provide one (path={metadata})")

        # default target is fire detection (0/1)
        self.target_names = target_names or ['fire']
        self.path_to_frames = path_to_frames
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """Returns the image and metadata

        Metadata contains the following information(not exhaustive list) :
            - fire(0/1) and clf_confidence(0/1)
            - x,y (float, float) and
            - Exploitable(True/False)"""
        path_to_frame = self.path_to_frames / self.metadata['imgFile'].iloc[index]
        observation = self.load_image(path_to_frame)

        if self.transform:
            observation = self.transform(observation)
        return observation, self._get_targets(index)

    def _get_targets(self, index):
        """Provide targets listed in target_names in metadata as Tensors

        Non-exhaustive values that can be found in self.target_names:
            ['fire', 'clf_confidence', 'loc_confidence', 'x', 'y']
        """
        return torch.from_numpy(self.metadata[self.target_names].iloc[index].values)


class WildFireSplitter:
    """Split one WildFireDataset into train, validation and test sets

    Three WildFireDataset instances will be created according the the given ratios.
    At this time, it is recommanded to transmit the transforms for each set.
    To avoid leakage in test and val test, splitting preserve fire_id consistency
    (ie: train set will only contain frames recorded before the ones contained in test set)

    Parameters
    ---------
    ratios: dict
        ratios (0 < float 1) corresponding to the number of frames to fill
        the Train, Val and Test set. it should have 'train', 'val' and 'test' keys.
        Example: {'train': 0.8, 'val':0.1, 'test': 0.1}

    algorithm: str, default='auto'
        Strategy to use to split the dataset. For now only 'auto' is implemented.

    Attributes
    ----------
    splits: dict
        Dictionnary containing the Splits(datasets) newly created by fit() method.
        It will have 'train', 'val' and 'test' keys.
        Example: {'train': WildFireDataset(), 'val':WildFireDataset(), 'test': WildFireDataset()

    wildfire: WildFireDataset, default=None
        Wildfire dataset to split

    Example
    -------
    wildfire = WildFireDataset(metadata='wildfire.csv', path_to_frames=path_to_frames)

    ratios = {'train': 0.7, 'val': 0.15, 'test':0.15}
    splitter =  WildFireSplitter(ratios)
    splitter.fit(wildfire)

    splitter/n_samples_ # {'train': 700, 'val': 147, 'test': 127}

    wildfire_loader_train = DataLoader(splitter.train, batch_size=64, shuffle=True)
    wildfire_loader_val = DataLoader(splitter.val, batch_size=64, shuffle=True)
    wildfire_loader_test = DataLoader(splitter.test, batch_size=64, shuffle=True)
    """
    def __init__(self, ratios, transforms=None, algorithm='auto', seed=42):
        self.seed = seed
        np.random.seed(seed)

        # Check ratios summed to one
        ratio_sum = sum((ratio for ratio in ratios.values()))
        if abs(ratio_sum - 1.) > 10e-4:
            raise ValueError(f"Ratio sum inconsistent. It should be unitary.\n"
                             f"Values found:"
                             f" Train({ratios['train']}) + Val({ratios['val']}) + Test({ratios['test']})"
                             f" = {ratio_sum} ≠ 1")

        self.ratios = ratios
        self.transforms = transforms or {'train': None, 'val': None, 'test': None}
        self.algorithm = algorithm

        # dict for datasets
        self.splits = {'train': None, 'val': None, 'test': None}
        self.wildfire = None

    # Some syntactic sugar
    @property
    def train(self):
        return self.splits['train']

    @property
    def val(self):
        return self.splits['val']

    @property
    def test(self):
        return self.splits['test']

    def fit(self, wildfire):
        """Split the wildfire dataset according to the given ratios.

        Set splits attribute
        Set also estimated posterior ratio(ratio_train_, ratio_val_ and ratio_test_)
        Because split is randomly done
        """
        self.wildfire = wildfire
        # Some checks first
        if wildfire.metadata['fire_id'].nunique() != wildfire.metadata['fire_id'].max() + 1:
            warnings.warn(f"Inconsistent Fire Labeling. Maybe try to label the fire again\n"
                          f"Distinct values of ids({wildfire.metadata['fire_id'].nunique()}"
                          f" ≠ {wildfire.metadata['fire_id'].max() + 1})", Warning)

        if self.algorithm != 'auto':
            raise ValueError(f"Algorithm {self.algorithm} is unknown. Only 'auto' available for now")
        else:
            self._strategy = ExhaustSplitStrategy

        # Let's split
        strategy = self._strategy()
        dataframes = strategy.split(wildfire, self.ratios, self.seed)
        self.set_splits(dataframes)

    def set_splits(self, dataframes):
        """Instantiate the Split as WildFireDataset and define the estimated parameters

        Parameters
        ----------
        dataframes: dict
            Dict containing the dataframes to feed the datasets corresponding to each split.
            It should have 'train', 'val' and 'test' as keys.
        """
        for set_ in ['train', 'val', 'test']:
            self.splits[set_] = WildFireDataset(metadata=dataframes[set_],
                                                path_to_frames=self.wildfire.path_to_frames,
                                                target_names=self.wildfire.target_names,
                                                transform=self.transforms[set_])

        # Determine estimated(posterior) parameters
        self.n_samples_ = {set_: len(self.splits[set_]) for set_ in ['train', 'val', 'test']}
        self.ratios_ = {set_: (self.n_samples_[set_] / len(self.wildfire)) for set_ in ['train', 'val', 'test']}


def computeSubSet(metadata, frame_per_seq, probTh=None):
    """This function computes a subset of the dataset, it extracts frame_per_seq consecutive frames for each sequence.

    Parameters
    ----------
        metadata: Pandas.DataFrame
        metadata of the WilFireDataset

    frame_per_seq: int
        frame per sequence to take

    frame_per_seq: float
        between 0 and 1, percentage of non-fire data to take twice in order to equalize the dataset

    probTh: float , optional
        The data set contains many more frames classified 'fire' than 'not fire', this parameter
        allows to equalize the dataset. For each 'not fire' sequence, we draw a random number if
        it is greater than probTh we double the number of frames used for this sequence

    Example
    -------
    metadataSS = computeSubSet(metadata, 2)
    wildfireSS = WildFireDataset(metadata=metadataSS, path_to_frames=path_to_frames)
    """
    if not isinstance(metadata, pd.DataFrame):
        try:
            metadata = pd.read_csv(metadata)
        except (ValueError, FileNotFoundError):
            raise ValueError(f"Invalid path to CSV containing metadata. Please provide one (path={metadata})")

    metadata.index = np.arange(len(metadata))
    imgs = metadata['imgFile']
    # Define sequences numbers
    metadata.index = np.arange(len(metadata))
    meta = metadata[['exploitable', 'fire', 'sequence', 'clf_confidence', 'loc_confidence',
                                    'x', 'y', 't', 'stateStart', 'stateEnd', 'fire_id', 'fBase']]
    meta = meta.drop_duplicates()
    meta['seq'] = np.arange(len(meta))
    metadata = pd.merge(metadata, meta, on=['exploitable', 'fire', 'sequence', 'clf_confidence',
                                                           'loc_confidence', 'x', 'y', 't', 'stateStart', 'stateEnd',
                                                           'fire_id', 'fBase'], how='inner')
    # Get unique list of sequences
    seq = metadata['seq']
    my_set = set(seq)
    uniqueSEQ = list(my_set)
    random.shuffle(uniqueSEQ)

    SubSetImgs = []
    SubSetImgsEq = []
    cryptogen = SystemRandom()
    for seU in uniqueSEQ:
        # For each sequence get a subSample of frame_per_seq frames
        nn = [imgs[i] for i, se in enumerate(seq) if se == seU]
        if(len(nn) > frame_per_seq):
            nn = random.sample(nn, frame_per_seq)
        nb = [float(frame.split("frame", 1)[1].split(".", 1)[0]) for frame in nn]
        nb, nn = (list(t) for t in zip(*sorted(zip(nb, nn))))
        SubSetImgs += nn
        # Equalize the dataset adding not_fire frames
        if probTh is not None:
            if(metadata[metadata['seq'] == seU]['fire'].values[0] == 0 and
               cryptogen.random() < probTh):
                nn = [imgs[i] for i, se in enumerate(seq) if se == seU]
                if(len(nn) > frame_per_seq):
                    nn = random.sample(nn, frame_per_seq)
                nb = [float(frame.split("frame", 1)[1].split(".", 1)[0]) for frame in nn]
                nb, nn = (list(t) for t in zip(*sorted(zip(nb, nn))))
                SubSetImgsEq += nn

    # Insert randomly the extra frames in the dataset
    if probTh is not None:
        for i in range(0, len(SubSetImgsEq), 2):
            idx = cryptogen.randint(0, len(SubSetImgs) - 2) // 2 * 2
            SubSetImgs.insert(idx, SubSetImgsEq[i + 1])
            SubSetImgs.insert(idx, SubSetImgsEq[i])

    # Create metadta Subset
    index = [i for i, im in enumerate(metadata['imgFile'].values) if im in SubSetImgs]

    return metadata.iloc[index]
