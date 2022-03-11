import logging

import numpy as np
import torch

from gradflow.grad_engine import Variable

log = logging.getLogger()


def test_add_result():
    variable_one = Variable(np.array([10, 20]))
    variable_two = Variable(np.array([30, 40]))
    
    result = variable_one + variable_two
    expected_result = np.array([40, 60])
    
    assert np.all(result.data == expected_result)


def test_add_grad():
    variable_one = Variable(np.array([10, 20]))
    variable_two = Variable(np.array([30, 40]))
    result = variable_one + variable_two
    result.backward()

    torch_variable_one = torch.tensor([10, 20], dtype=torch.float32, requires_grad=True)
    torch_variable_two = torch.tensor([30, 40], dtype=torch.float32, requires_grad=True)
    torch_result = torch_variable_one + torch_variable_two
    torch_result.backward([torch.tensor([1, 1])])

    print(torch_variable_one.grad)
    print(torch_variable_two.grad)

    print(variable_one.grad)
    print(variable_two.grad)


def test_multiply_result():
    pass


def test_multiply_grad():
    pass



if __name__ == "__main__":
    test_add_result()
    test_add_grad()

    test_multiply_result()
    test_multiply_grad()