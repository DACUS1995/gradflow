import unittest

import numpy as np

from gradflow import Variable
from gradflow.utils import no_grad


class TestUtils(unittest.TestCase):
    def test_no_grad(self):
        variable_one = Variable(np.array([10, 20]), requires_grad=True)
        
        with no_grad():
            variable_two = Variable(variable_one)
            variable_three = variable_one + variable_two

        variable_four = variable_one + variable_two

        self.assertTrue(variable_two.requires_grad == False)
        self.assertTrue(variable_three.requires_grad == False)
        self.assertTrue(variable_four.requires_grad == True)
