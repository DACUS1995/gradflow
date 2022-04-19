import unittest

import numpy as np
import torch

from gradflow.grad_engine import Variable
from gradflow.activations.common import relu


class TestCommonActivations(unittest.TestCase):
    def test_relu(self):
        input = Variable(np.array([-10, 10], dtype=np.float32), requires_grad=True)
        output = relu(input)
        output.backward(np.array([1, 1]))

        torch_input = torch.tensor([-10, 10], dtype=torch.float32, requires_grad=True)
        torch_output = torch.nn.functional.relu(torch_input)
        torch_output.backward(torch.tensor([1, 1]))

        print(input.grad)
        print(torch_input.grad.numpy())

        self.assertTrue(np.all(output.data == np.array([0, 10])))
        self.assertTrue(np.all(output.data == torch_output.detach().numpy()))
        self.assertTrue(np.all(input.grad == torch_input.grad.numpy()))

if __name__ == "__main__":
    unittest.main()