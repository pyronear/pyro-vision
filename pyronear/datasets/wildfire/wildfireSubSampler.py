import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from pyronear.datasets.utils import VisionMixin
from pyronear.datasets.wildfire import ExhaustSplitStrategy
import random
import PIL


class wildfireSubSampler(Dataset):
    """WildFireSubSampler dataset that can be fed to a torch model

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

    frame_per_seq: int
        Number of frames to take for each sequence

    transform: object, optional
        Transformations to apply to the frames (ie: torchvision.transforms)
    """

    def __init__(self, metadata, path_to_frames, frame_per_seq, target_names=None, transform=None):
        if isinstance(metadata, pd.DataFrame):
            self.metadata = metadata
        else:
            try:
                self.metadata = pd.read_csv(metadata)
            except (ValueError, FileNotFoundError):
                raise ValueError(f"Invalid path to CSV containing metadata. Please provide one (path={metadata})")

        self.path_to_frames = Path(path_to_frames)
        self.target_names = target_names or ['fire']
        self.transform = transform
        self.frame_per_seq = frame_per_seq
        self.metadata.index = np.arange(len(self.metadata))
        #Get unique list of sequences
        self.seq = self.metadata['seq']
        my_set = set(self.seq)
        self.uniqueSEQ = list(my_set)
        random.shuffle(self.uniqueSEQ)
        self.imgs = self.metadata['imgFile']
        self.SubSetImgs = []
        self.probTh = None

    def computeSubSet(self, probTh=None):
        """This function computes the subset, it extracts frame_per_seq frames for each sequence.

        Parameters
        ----------
        probTh: float , optional
            The data set contains many more frames classified 'fire' than 'not fire', this parameter
            allows to equalize the dataset. For each 'not fire' sequence, we draw a random number if
            it is greater than probTh we double the number of frames used for this sequence
        """
        meta = self.metadata
        self.SubSetImgs = []
        self.probTh = probTh
        SubSetImgsEq = []

        for seU in self.uniqueSEQ:
            #For each sequence get a subSample of frame_per_seq frames
            nn = [self.imgs[i] for i, se in enumerate(self.seq) if se == seU]
            if(len(nn) > self.frame_per_seq):
                nn = random.sample(nn, self.frame_per_seq)
            nb = [float(frame.split("frame", 1)[1].split(".", 1)[0]) for frame in nn]
            nb, nn = (list(t) for t in zip(*sorted(zip(nb, nn))))
            self.SubSetImgs += nn
            #Equalize the dataset adding not_fire frames
            if probTh is not None:
                if(meta[meta['seq'] == seU]['fire'].values[0] == 0 and random.random() > probTh):
                    nn = [self.imgs[i] for i, se in enumerate(self.seq) if se == seU]
                    if(len(nn) > self.frame_per_seq):
                        nn = random.sample(nn, self.frame_per_seq)
                    nb = [float(frame.split("frame", 1)[1].split(".", 1)[0]) for frame in nn]
                    nb, nn = (list(t) for t in zip(*sorted(zip(nb, nn))))
                    SubSetImgsEq += nn

        #Insert randomly the extra frames in the dataset
        if probTh is not None:
            for i in range(0, len(SubSetImgsEq), 2):
                idx = random.randint(0, len(self.SubSetImgs) - 2) // 2 * 2
                self.SubSetImgs.insert(idx, SubSetImgsEq[i + 1])
                self.SubSetImgs.insert(idx, SubSetImgsEq[i])

    def __len__(self):
        return len(self.SubSetImgs)

    def __getitem__(self, index):
        img = self.SubSetImgs[index]
        #get the corresponding metadata
        meta = self.metadata[self.metadata['imgFile'] == img]
        path_to_frame = self.path_to_frames / str(meta['imgFile'].values[0])
        observation = PIL.Image.open(path_to_frame)

        if self.transform:
            observation = self.transform(observation)

        #return observation, meta.values
        return observation, meta[self.target_names].values[0]


class WildFireSplitter:
    """Split one wildfireSubSampler into train, validation and test sets

    Three wildfireSubSampler instances will be created according the the given ratios.
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
    wildfire = wildfireSubSampler(metadata=metadata,path_to_frames=path_to_frames,frame_per_seq=K,target_names=['fire'])
    wildfire.computeSubSet(0)
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

    def fit(self, wildfireSS):
        """Split the wildfire dataset according to the given ratios.

        Set splits attribute
        Set also estimated posterior ratio(ratio_train_, ratio_val_ and ratio_test_)
        Because split is randomly done
        """
        self.wildfireSS = wildfireSS

        if self.algorithm != 'auto':
            raise ValueError(f"Algorithm {self.algorithm} is unknown. Only 'auto' available for now")
        else:
            self._strategy = ExhaustSplitStrategy

        # Let's split
        strategy = self._strategy()
        dataframes = strategy.split(self.wildfireSS, self.ratios, self.seed)
        self.set_splits(dataframes)

    def set_splits(self, dataframes):
        """Instantiate the Split as wildfireSubSampler and define the estimated parameters

        Parameters
        ----------
        dataframes: dict
            Dict containing the dataframes to feed the datasets corresponding to each split.
            It should have 'train', 'val' and 'test' as keys.
        """
        for set_ in ['train', 'val', 'test']:
            self.splits[set_] = wildfireSubSampler(metadata=dataframes[set_],
                                                   path_to_frames=self.wildfireSS.path_to_frames,
                                                   target_names=self.wildfireSS.target_names,
                                                   frame_per_seq=self.wildfireSS.frame_per_seq,
                                                   transform=self.transforms[set_])

            self.splits[set_].computeSubSet(self.wildfireSS.probTh)

        # Determine estimated(posterior) parameters
        self.n_samples_ = {set_: len(self.splits[set_]) for set_ in ['train', 'val', 'test']}
        self.ratios_ = {set_: (self.n_samples_[set_] / len(self.wildfireSS)) for set_ in ['train', 'val', 'test']}
