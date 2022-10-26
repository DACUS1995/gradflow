import unittest

import numpy as np

from gradflow.grad_engine import Variable
from gradflow.layers.linear import Linear


class TestLinearLayer(unittest.TestCase):
    def test_linear_layer_forward(self):
        input = Variable(np.array([10, 20]))

        linear_model = Linear(in_size=2, out_size=2)
        linear_model.weights.data.data = np.ones_like(linear_model.weights.data.data)
        linear_model.b.data.data = np.ones_like(linear_model.b.data.data)

        output = linear_model(input)

        self.assertTrue(np.all(output.data == np.array([31, 31])))

if __name__ == "__main__":
    unittest.main()
