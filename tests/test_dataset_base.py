import pytest
import numpy as np
from numpy.random import RandomState

from sygnal.dataset.base import RefGameDatasetAbstractBase
from sygnal.dataset.base import RefGameSamplerBase


class RefGameDatasetTestBase:
    PRE_GEN_DATASET_DEFAULT_SIZE=10
    DATASETS_DEFAULT_SEED=42

    @pytest.fixture
    def pre_generated(self, dataset_cls):
        size = self.PRE_GEN_DATASET_DEFAULT_SIZE
        seed = self.DATASETS_DEFAULT_SEED
        return dataset_cls.pre_generate(size, seed)

    @pytest.fixture
    def lazily_generated(self, dataset_cls):
        seed = self.DATASETS_DEFAULT_SEED
        lazily_generated = dataset_cls()
        rnd = RandomState(seed)
        lazily_generated.random_state = rnd.get_state()
        return lazily_generated


class TestRefGameDatasetAbstractBase(RefGameDatasetTestBase):

    @pytest.fixture
    def dataset_cls(self, mocker):
        """Provide a fixture that patches the abstract base
        class with a dummy method allowing us to test it.
        """
        mocker.patch.multiple(RefGameDatasetAbstractBase,
                              __abstractmethods__=set())
        return RefGameDatasetAbstractBase

    def test_generate(self, dataset_cls):
        """Checks that the internal method for specifying the
        dataset length works as expected."""
        size = 10
        instance = dataset_cls()
        instance._generate(size)

        assert len(instance) == size, 'Sample list has not the required size!'

    def test_pre_generate(self, pre_generated):
        """Test that pre-generated data-set initialization values
        results a correct internal initialization.
        """

        assert len(pre_generated) == self.PRE_GEN_DATASET_DEFAULT_SIZE, \
            'Sample list has not the required size!'
        assert pre_generated.random_state[1][0] == self.DATASETS_DEFAULT_SEED, \
            'Seed of RandomState was set incorrectly.'

    def test_getitem_only_pre_gen(self, pre_generated, lazily_generated):
        """It should be impossible to get an item from a non pre-generated
        data-set.
        """
        # No TypeError
        pre_generated[5]

        # Raises TypeError
        with pytest.raises(TypeError):
            lazily_generated[5]

    def test_len_only_pre_gen(self, pre_generated, lazily_generated):
        """The length of a generator should rise an exception."""
        # No TypeError
        assert len(pre_generated) == self.PRE_GEN_DATASET_DEFAULT_SIZE

        # Raises TypeError
        with pytest.raises(TypeError):
            len(lazily_generated)


class RefGameDatasetImplTestBase(RefGameDatasetTestBase):
    """Utility class to be inherited from other data-set classes
    to be automatically tested.
    """

    # Inherit this class and change this variable to point
    # to the target data-set to test.
    DATASET_CLS=None

    @pytest.fixture
    def dataset_cls(self):
        assert self.DATASET_CLS is not None
        assert issubclass(self.DATASET_CLS, RefGameDatasetAbstractBase)
        return self.DATASET_CLS

    def test_iter_equal_pre_gen(self, dataset_cls):
        """Given the same random-number-generator state,
        both pre-generated and lazily generated data-set should result
        in the same samples."""
        size = self.PRE_GEN_DATASET_DEFAULT_SIZE
        seed = self.DATASETS_DEFAULT_SEED

        pre_generated = dataset_cls.pre_generate(size, seed)
        lazily_generated = dataset_cls()
        rnd = RandomState(seed)
        lazily_generated.random_state = rnd.get_state()

        for i, sample in enumerate(lazily_generated):
            if i >= size - 1:
                break
            print(self.DATASET_CLS)
            assert sample == pre_generated[i], \
                f'Samples at iteration {i} don\'t match!'


class TestRefGameSamplerBase:

    def test_len(self):
        fewer_sampler = RefGameSamplerBase(0, 5)
        more_sampler = RefGameSamplerBase(0, 15)

        assert len(fewer_sampler) == 5
        assert len(more_sampler) == 15

    def test_iter(self):
        ndistractors = 3
        nsamples = 100

        sampler = RefGameSamplerBase(ndistractors, nsamples)

        for i, s in enumerate(sampler):
            assert len(s) == ndistractors + 1, \
                f'Expect sample to have {ndistractors} + 1 values.'
            assert s[0] not in s[1:], \
                f'Expect distractors to be different from target.' \
                f'But found target: {s[0]} and distractors {s[1:]}'

        assert i == nsamples - 1, 'Wrong number of iterations.'
