from typing import Union
import unittest

import numpy as np
import torch

from gradflow.grad_engine import Variable
from gradflow.optimizers import NaiveSGD
from gradflow.layers.module import Module
from gradflow.optimizers import Adam
from gradflow.optimizers.optimizers import BaseOptimizer


_default_params = {
    "lr": 3e-4
} 


class TestOptimizers(unittest.TestCase):
    def test_optimizers_zero_grad(self):
        variable_one = Variable(np.array([10, 20], dtype=np.float32), requires_grad=True)
        variable_two = Variable(np.array([30, 40], dtype=np.float32), requires_grad=True)
        result = variable_one + variable_two
        result.backward(np.ones_like(result.data))

        optimizer = NaiveSGD([variable_one, variable_two], **_default_params)
        optimizer.zero_grad()

        self.assertTrue(np.all(variable_one.grad == np.zeros_like(variable_one.grad)))
        self.assertTrue(np.all(variable_two.grad == np.zeros_like(variable_two.grad)))
        

    def test_naive_sgd(self):
        torch_variable_one = torch.tensor([10, 20], dtype=torch.float32, requires_grad=True)
        torch_variable_two = torch.tensor([30, 40], dtype=torch.float32, requires_grad=True)
        torch_sgd = torch.optim.SGD(params=[torch_variable_one],**_default_params)
        self.perform_optimiser_step(torch_sgd, torch_variable_one, torch_variable_two)

        gradflow_variable_one = Variable(np.array([10, 20], dtype=np.float32), requires_grad=True)
        gradflow_variable_two = Variable(np.array([30, 40], dtype=np.float32), requires_grad=True)
        gradflow_sgd = NaiveSGD(parameters=[gradflow_variable_one], **_default_params)
        self.perform_optimiser_step(gradflow_sgd, gradflow_variable_one, gradflow_variable_two)

        self.assertTrue(np.array_equal(torch_variable_one.detach().numpy(), gradflow_variable_one.data.data))


    def test_adam(self):
        torch_variable_one = torch.tensor([10, 20], dtype=torch.float32, requires_grad=True)
        torch_variable_two = torch.tensor([30, 40], dtype=torch.float32, requires_grad=True)
        torch_adam = torch.optim.Adam(params=[torch_variable_one], **_default_params)
        # Performing two steps
        self.perform_optimiser_step(torch_adam, torch_variable_one, torch_variable_two)
        self.perform_optimiser_step(torch_adam, torch_variable_one, torch_variable_two)

        gradflow_variable_one = Variable(np.array([10, 20], dtype=np.float32), requires_grad=True)
        gradflow_variable_two = Variable(np.array([30, 40], dtype=np.float32), requires_grad=True)
        gradflow_adam = Adam(parameters=[gradflow_variable_one], **_default_params)
        # Performing two steps
        self.perform_optimiser_step(gradflow_adam, gradflow_variable_one, gradflow_variable_two)
        self.perform_optimiser_step(gradflow_adam, gradflow_variable_one, gradflow_variable_two)

        self.assertTrue(np.array_equal(torch_variable_one.detach().numpy(), gradflow_variable_one.data.data))


    def perform_optimiser_step(
        self, 
        optimizer: Union[torch.optim.Optimizer, BaseOptimizer], 
        variable_one: Union[torch.Tensor, Variable], 
        variable_two: Union[torch.Tensor, Variable]
    ):
        is_torch = isinstance(optimizer, torch.optim.Optimizer)
        result = variable_one + variable_two
        
        if is_torch:
            result.backward(torch.ones_like(variable_one))
        else:
            result.backward(np.ones_like(variable_one.data.data))

        optimizer.step()
        

if __name__ == "__main__":
    unittest.main()
