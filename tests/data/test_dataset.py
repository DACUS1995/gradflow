import unittest

import numpy as np
import torch

from gradflow.grad_engine import Variable
from gradflow.data import Dataset


class TestCommonActivations(unittest.TestCase):
    def test_dataset_indexing(self):
        features = np.random.rand(10, 5)
        labels = np.ones((10))
        batch_size = 2

        dataset = Dataset(features, labels, batch_size)

        self.assertTrue(np.all(dataset[0][0] == features[0:batch_size]))
        self.assertTrue(np.all(dataset[0][1] == labels[0:batch_size]))
        self.assertTrue(dataset[0].shape[0] == batch_size)


if __name__ == "__main__":
    unittest.main()