"""Base classes for dataset geneartion.

This module provides a set of classes meant
to be specialized (inherited).
"""

from abc import ABC
from abc import abstractmethod

from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data import Sampler

import numpy as np
from numpy.random import RandomState


class RefGameDatasetAbstractBase(IterableDataset, ABC):
    """Base class that defines a referential game data-set.
    This class provides some simple boilerplate to handle
    infinite datasets and save pre-generated ones. All at the cost
    of implementing _generate_sample method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._samples = None
        self._rng = RandomState(None)

    @classmethod
    def pre_generate(cls, size, seed=None):
        """Takes care of generating a fixed size dataset."""
        dataset = cls()
        dataset._rng = RandomState(seed)
        dataset._generate(size)
        return dataset

    def _generate(self, size):
        self._samples = [self._generate_sample() for i in range(size)]

    @classmethod
    def load(self, path):
        """Loads a dataset located in a certain path."""
        raise NotImplementedError

    def save(self, path):
        """Saves the current dataset in a specific path in some default way.
        (i.e. pickles).
        NOTE(lromor): If necessary we could define pickle classes
        to better handle how to pickle the dataset downstream.
        For now we don't have any fancy requirement.
        """
        raise NotImplementedError

    @property
    def random_state(self):
        """Returns the current random state. The returned object can be useful
        to restore the random number generator to some specific state."""
        return self._rng.get_state()

    @random_state.setter
    def random_state(self, state):
        """Sets the random number generator with the provided state."""
        self._rng.set_state(state)

    @abstractmethod
    def _generate_sample(self):
        pass

    def __len__(self):
        if self._samples is not None:
            return len(self._samples)
        else:
            raise TypeError(
                'Datasets without pregenerated samples have no/infinite length.')

    def __getitem__(self, key):
        if self._samples is not None:
            return self._samples[key]
        else:
            raise TypeError("The current dataset instance is not a "
                            "pregenerated dataset hence it's not subscriptable.")

    def __iter__(self):
        """Returns an iterator that let's you lazily loop through the dataset.
        TODO: support multiple workers using torch.utils.data.get_worker_info()
        https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        # If the dataset is stored
        # let's return an iterator from it
        if self._samples:
            return iter(self._samples)

        # Otherwise let's build a generator
        # that can generate infinite samples.
        def data_gen():
            while True:
                yield self._generate_sample()

        return data_gen()


class RefGameSamplerBase(Sampler):
    """Basic sampler that iterates through (shuffled) indicies
    and draws disctractors without overlap for each sample.
    """

    def __init__(self, ndistractors, nsamples, shuffle=False):
        """Constructor.

        Args:
            - ndistractors: number of disctractor indicies for each sample
            - nsamples: total number of samples to be generated
            - shuffle: wether to shuffle list
        """
        self.ndistractors = ndistractors
        self.nsamples = nsamples
        self.shuffle = shuffle

    def __iter__(self):
        indices = []

        if self.shuffle:
            targets = torch.randperm(self.n).tolist()
        else:
            targets = list(range(self.nsamples))

        for t in targets:
            # target + distractors
            arr = np.zeros(self.ndistractors + 1, dtype=int)
            arr[0] = t

            choices = list(range(self.nsamples))
            choices.remove(t)
            distractors = np.random.choice(choices,
                                           size=self.ndistractors,
                                           replace=False)
            arr[1:] = distractors

            indices.append(arr)

        return iter(indices)

    def __len__(self):
        return self.nsamples
