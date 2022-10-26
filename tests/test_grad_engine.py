import logging
import unittest

import numpy as np
import torch

from gradflow.grad_engine import Variable

log = logging.getLogger()


class TestGradEngine(unittest.TestCase):
    def test_variable_copy_init(self):
        variable_one = Variable(np.array([10, 20]))
        variable_two = Variable(variable_one)

        self.assertTrue(variable_one.data is variable_two.data)

    
    def test_wrong_data_type_init(self):
        with self.assertRaises(TypeError):
            Variable([1,2])


    def test_add_result(self):
        variable_one = Variable(np.array([10, 20]))
        variable_two = Variable(np.array([30, 40]))
        
        result = variable_one + variable_two
        expected_result = np.array([40, 60])
        
        self.assertTrue(np.all(result.data == expected_result))
    
    
    def test_add_grad(self):
        variable_one = Variable(np.array([10, 20], dtype=np.float32), requires_grad=True)
        variable_two = Variable(np.array([30, 40], dtype=np.float32), requires_grad=True)
        result = variable_one + variable_two
        result.backward(np.ones_like(result.data.data))
    
        torch_variable_one = torch.tensor([10, 20], dtype=torch.float32, requires_grad=True)
        torch_variable_two = torch.tensor([30, 40], dtype=torch.float32, requires_grad=True)
        torch_result = torch_variable_one + torch_variable_two
        torch_result.backward(torch.tensor([1, 1]))
    
        self.assertTrue(np.array_equal(torch_variable_one.grad.numpy(), variable_one.grad))
        self.assertTrue(np.array_equal(torch_variable_two.grad.numpy(), variable_two.grad))


    def test_substract_result(self):
        variable_one = Variable(np.array([10, 20]))
        variable_two = Variable(np.array([30, 40]))
        
        result = variable_one - variable_two
        expected_result = np.array([-20, -20])
        
        self.assertTrue(np.all(result.data == expected_result))


    def test_substract_grad(self):
        variable_one = Variable(np.array([10, 20], dtype=np.float32), requires_grad=True)
        variable_two = Variable(np.array([30, 40], dtype=np.float32), requires_grad=True)
        result = variable_one - variable_two
        result.backward(np.ones_like(result.data.data))
    
        torch_variable_one = torch.tensor([10, 20], dtype=torch.float32, requires_grad=True)
        torch_variable_two = torch.tensor([30, 40], dtype=torch.float32, requires_grad=True)
        torch_result = torch_variable_one - torch_variable_two
        torch_result.backward(torch.tensor([1, 1]))

        self.assertTrue(np.array_equal(torch_variable_one.grad.numpy(), variable_one.grad))
        self.assertTrue(np.array_equal(torch_variable_two.grad.numpy(), variable_two.grad))


    def test_vector_multiply_result(self):
        variable_one = Variable(np.array([10, 20]))
        variable_two = Variable(np.array([30, 40]))
        
        result = variable_one @ variable_two.T()
        expected_result = np.array([1100])
        
        self.assertTrue(np.all(result.data.data == expected_result))


    def test_vector_multiply_grad(self):
        variable_one = Variable(np.array([10, 20], dtype=np.float32), requires_grad=True)
        variable_two = Variable(np.array([30, 40], dtype=np.float32), requires_grad=True)
        result = variable_one @ variable_two.T()
        result.backward(np.ones_like(result.data.data))
    
        torch_variable_one = torch.tensor([10, 20], dtype=torch.float32, requires_grad=True)
        torch_variable_two = torch.tensor([30, 40], dtype=torch.float32, requires_grad=True)
        torch_result = torch_variable_one @ torch_variable_two.t()
        torch_result.backward()

        self.assertTrue(np.array_equal(torch_variable_one.grad.numpy(), variable_one.grad))
        self.assertTrue(np.array_equal(torch_variable_two.grad.numpy(), variable_two.grad))



if __name__ == "__main__":
    unittest.main()
