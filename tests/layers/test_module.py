import unittest

import numpy as np

from gradflow.grad_engine import Variable
from gradflow.layers.module import Module


class TestModule(unittest.TestCase):
    def test_module(self):
        parameter_one = Variable(np.array([10, 20]), requires_grad=True)
        parameter_two = Variable(np.array([30, 40]), requires_grad=True)
        module_one = Module()

        class ModuleTest(Module):
            def __init__(self) -> None:
                super().__init__()
                self.parameter_one = self.add_parameter(parameter_one)
                self.parameter_two = self.add_parameter(parameter_two)
                self.module_one = self.add_module(module_one)

        module_test = ModuleTest()

        self.assertTrue(len(module_test.parameters) == 2)
        self.assertTrue(len(module_test.modules) == 1)


    def test_module_parameters_cycle(self):
        pass


if __name__ == "__main__":
    unittest.main()