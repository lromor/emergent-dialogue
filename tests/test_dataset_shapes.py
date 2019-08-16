import pytest

from sygnal.dataset.shapes import ShapesImageDataset

from .test_dataset_base import RefGameDatasetImplTestBase


class TestShapesImageDataset(RefGameDatasetImplTestBase):
    DATASET_CLS=ShapesImageDataset
