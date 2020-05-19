# -*- coding: utf-8 -*-

import abc
import copy

import numpy as np


class SplitStrategy(metaclass=abc.ABCMeta):
    """Abstract Class to define Splitting strategies"""
    @abc.abstractmethod
    def split(self, dataset, ratios):
        """Method that should split the dataset and return the splits as a dataframes
        in a dict with the following keys:
        {'train': df_train, 'val': df_val, 'test': df_test}

        Parameters
        ----------
        dataset: instance of a dataset(ie: WildFireDataset)
            dataset to split into Train, Val and Test sets

        ratios: dict
            Ratios to use to split the dataset.
            Example: {'train': 0.7, 'val': 0.15, 'test':0.15}
        """


class ExhaustSplitStrategy(SplitStrategy):
    """Splitting strategy that split a dataset by exhausting fire ids"""
    @staticmethod
    def random_fire_ids_gen(n_samples, fire_id_to_size):
        """Generate fire_ids till they approximately encompass n_samples"""
        # While there is still samples to exhaust
        while n_samples > 0:
            # randomly yield a remaining fire_id
            random_fire_id = np.random.choice(list(fire_id_to_size.keys()))
            yield random_fire_id

            # Take the fire id and the matched samples out
            n_samples = n_samples - fire_id_to_size[random_fire_id]
            del fire_id_to_size[random_fire_id]

    def _get_fire_ids_for_one_split(self, n_samples):
        """Return list of fire_ids representing count of n_samples.
        For instance, returns [90, 118, 67] to match a 10% test ratio

        n_samples: Number of samples to fill the split
        """
        fire_ids = list(self.random_fire_ids_gen(n_samples, self._fire_id_to_size_to_exhaust))
        return fire_ids

    def split(self, dataset, ratios, seed=42):
        """Split the dataset in Train/Val/Test according to ratio set at init
        This strategy randomly exhausts the fire ids list
        so they fills the splits as respectfully to the given ratio as possible

        Note: So far, it has only been tested with WildFireDataset.
        """
        np.random.seed(seed)
        df = dataset.metadata  # alias for convenience (less verbose)

        n_samples_total = df.shape[0]
        n_samples_train = n_samples_total * ratios['train']
        n_samples_val = n_samples_total * ratios['val']
        #n_samples_test = n_samples_total - (n_samples_train + n_samples_val)

        # create hash table to exhaust: {fire_id: number of frames labeled with fire_id}
        self._fire_id_to_size = df.groupby('fire_id').size().to_dict()
        self._fire_id_to_size_to_exhaust = copy.deepcopy(self._fire_id_to_size)

        # Let's get
        if ratios['test'] > 0:
            fire_ids = {'train': self._get_fire_ids_for_one_split(n_samples_train),
                        'val': self._get_fire_ids_for_one_split(n_samples_val)}
            fire_ids['test'] = [id_ for id_ in self._fire_id_to_size
                                if id_ not in (fire_ids['train'] + fire_ids['val'])]
            # Finish exhaustion
            for fire_id_test in fire_ids['test']:
                del self._fire_id_to_size_to_exhaust[fire_id_test]

        else:
            fire_ids = {'train': self._get_fire_ids_for_one_split(n_samples_train)}
            fire_ids['val'] = [id_ for id_ in self._fire_id_to_size if id_ not in fire_ids['train']]
            # Finish exhaustion
            for fire_id_test in fire_ids['val']:
                del self._fire_id_to_size_to_exhaust[fire_id_test]
            fire_ids['test'] = []

        n_samples_remaining = len(self._fire_id_to_size_to_exhaust)
        if n_samples_remaining != 0:
            raise ValueError(f"Algorithm failing, {n_samples_remaining} samples not assigned to any split!")

        return {set_: df[df['fire_id'].isin(fire_ids[set_])] for set_ in ['train', 'val', 'test']}
