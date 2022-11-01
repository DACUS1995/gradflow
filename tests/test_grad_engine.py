import logging
import unittest

import numpy as np
import torch
import parameterized

from gradflow.grad_engine import Variable
from gradflow.data_containers import NumpyDataContainer, GPUDataContainer

log = logging.getLogger()


@parameterized.parameterized_class([{"device": "cpu"}, {"device": "gpu"}])
class TestGradEngine(unittest.TestCase):
    def test_variable_copy_init(self):
        variable_one = Variable(np.array([10, 20]), device=self.device)
        variable_two = Variable(variable_one, device=self.device)

        self.assertTrue(variable_one.data is variable_two.data)

    
    def test_wrong_data_type_init(self):
        with self.assertRaises(TypeError):
            Variable([1,2])


    def test_add_result(self):
        variable_one = Variable(np.array([10, 20]), device=self.device)
        variable_two = Variable(np.array([30, 40]), device=self.device)
        
        result = variable_one + variable_two
        expected_result = np.array([40, 60])
        
        self.assertTrue(np.all(result.data == expected_result))
    
    
    def test_add_grad(self):
        variable_one = Variable(np.array([10, 20], dtype=np.float32), requires_grad=True, device=self.device)
        variable_two = Variable(np.array([30, 40], dtype=np.float32), requires_grad=True, device=self.device)
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

    
    def test_device_to_data_container(self):
        variable_cpu = Variable(np.array([1,2]), device="cpu")
        variable_gpu = Variable(np.array([1,2]), device="gpu")

        self.assertTrue(isinstance(variable_cpu.data, NumpyDataContainer))
        self.assertTrue(isinstance(variable_gpu.data, GPUDataContainer))


    def test_device_transfer(self):
        variable_cpu = Variable(np.array([1,2]), device="cpu")
        variable_gpu = variable_cpu.to("gpu")
        self.assertTrue(variable_gpu.device == "gpu")
        self.assertTrue(isinstance(variable_gpu.data, GPUDataContainer))

        variable_gpu = variable_gpu.to("gpu")
        self.assertTrue(variable_gpu.device == "gpu")
        self.assertTrue(isinstance(variable_gpu.data, GPUDataContainer))

        variable_cpu = variable_gpu.to("cpu")
        self.assertTrue(variable_cpu.device == "cpu")
        self.assertTrue(isinstance(variable_cpu.data, NumpyDataContainer))



if __name__ == "__main__":
    unittest.main()
