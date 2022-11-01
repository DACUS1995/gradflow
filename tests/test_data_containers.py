import logging
import unittest

import numpy as np

from gradflow.data_containers import NumpyDataContainer, GPUDataContainer

log = logging.getLogger()


class TestGradEngine(unittest.TestCase):
	def test_cpu_variable_copy_init(self):
		container_one = NumpyDataContainer(np.array([10, 20]))
		container_two = NumpyDataContainer(container_one)

		self.assertTrue(container_one.data is container_two.data)

	def test_gpu_variable_copy_init(self):
		container_one = GPUDataContainer(np.array([10, 20]))
		container_two = GPUDataContainer(container_one)

		self.assertTrue(container_one.data is container_two.data)

