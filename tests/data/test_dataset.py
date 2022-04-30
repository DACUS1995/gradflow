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

        self.assertTrue(dataset[0][0].data.shape[0] == batch_size)
        self.assertTrue(np.all(dataset[0][0].data == features[0:batch_size]))
        self.assertTrue(np.all(dataset[0][1].data == labels[0:batch_size]))
        
        with self.assertRaises(IndexError):
            dataset[10]
        

if __name__ == "__main__":
    unittest.main()