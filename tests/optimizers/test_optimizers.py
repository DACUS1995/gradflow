import unittest

import numpy as np

from gradflow.grad_engine import Variable
from gradflow.optimizers import NaiveSGD
from gradflow.layers.module import Module


class TestOptimizers(unittest.TestCase):
    def test_optimizers_zero_grad(self):
        variable_one = Variable(np.array([10, 20], dtype=np.float32), requires_grad=True)
        variable_two = Variable(np.array([30, 40], dtype=np.float32), requires_grad=True)
        result = variable_one + variable_two
        result.backward(np.ones_like(result.data))

        optimizer = NaiveSGD([variable_one, variable_two])
        optimizer.zero_grad()

        self.assertTrue(np.all(variable_one.grad == np.zeros_like(variable_one.grad)))
        self.assertTrue(np.all(variable_two.grad == np.zeros_like(variable_two.grad)))


if __name__ == "__main__":
    unittest.main()