import unittest
from unittest.mock import patch

import numpy as np
from numpy.random import RandomState

from sygnal.dataset.base import RefGameDatasetBase, RefGameSamplerBase


@patch.multiple(RefGameDatasetBase, __abstractmethods__=set())
class RefGameDatasetBaseTest(unittest.TestCase):

    def test_generate(self):
        size = 10
        instance = RefGameDatasetBase()
        instance._generate(size)

        self.assertEqual(len(instance), size,
                         'Sample list has not the required size!')

    def test_pre_generate(self):
        size = 10
        seed = 42

        pre_gen = RefGameDatasetBase.pre_generate(size, seed)

        self.assertEqual(len(pre_gen), size,
                         'Sample list has not the required size!')
        self.assertEqual(pre_gen.random_state[1][0], seed,
                         'Seed of RandomState was set incorrectly.')

    def test_iter_equal_pre_gen(self):
        size = 10
        seed = 42

        pre_gen = RefGameDatasetBase.pre_generate(size, seed)

        dynamic = RefGameDatasetBase()
        rnd = RandomState(seed)
        dynamic.random_state = rnd.get_state()

        for i, sample in enumerate(dynamic):
            if i >= size - 1:
                break
            self.assertEqual(
                sample, pre_gen[i], f'Samples at iteration {i} don\'t match!')

    def test_getitem_only_pre_gen(self):
        size = 10

        pre_gen = RefGameDatasetBase.pre_generate(size)
        dynamic = RefGameDatasetBase()

        # No TypeError
        pre_gen[5]

        # Raises TypeError
        with self.assertRaises(TypeError):
            dynamic[5]

    def test_len_only_pre_gen(self):
        size = 10

        pre_gen = RefGameDatasetBase.pre_generate(size)
        dynamic = RefGameDatasetBase()

        # No TypeError
        self.assertEqual(len(pre_gen), size)

        # Raises TypeError
        with self.assertRaises(TypeError):
            len(dynamic)


class RefGameSamplerBaseTest(unittest.TestCase):

    def test_len(self):
        fewer_sampler = RefGameSamplerBase(0, nsamples=5)
        more_sampler = RefGameSamplerBase(0, nsamples=15)

        self.assertEqual(len(fewer_sampler), 5)
        self.assertEqual(len(more_sampler), 15)

    def test_iter(self):
        ndistractors = 3
        nsamples = 100

        sampler = RefGameSamplerBase(ndistractors, nsamples)

        for i, s in enumerate(sampler):
            self.assertEqual(len(s), ndistractors + 1,
                             f'Expect sample to have {ndistractors} + 1 values.')
            self.assertNotIn(s[0], s[1:],
                             f'Expect distractors to be different from target.' +
                             f'But found target: {s[0]} and distractors {s[1:]}')

        self.assertEqual(i, nsamples - 1, 'Wrong number of iterations.')
