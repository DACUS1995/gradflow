from __future__ import annotations
from typing import List, Tuple, Dict, Union

import numpy as np


class Variable:
	def __init__(self, data: Variable | np.ndarray, parents: Tuple = ()) -> None:
		if isinstance(data, self.__class__):
			data = data.data
		elif not isinstance(data, np.ndarray):
			raise TypeError("The data type provided is not supported")

		self.data = data
		self.grad = 0
		self.parents = parents
		self._back_grad_fn = lambda: None


	def __add__(self, other: Variable) -> Variable:
		result = self.data + other.data
		variable = Variable(result, parents=(self, other))

		def _back_grad_fn():
			self.grad += variable.grad
			other.grad += variable.grad

		variable._back_grad_fn = _back_grad_fn
		return variable


	def __mul__(self, other: Variable) -> Variable:
		self.grad = other.data
		other.grad = self.data
		result = np.matmul(self.data, other.data)
		variable = Variable(result, parents=(self, other))

		def _back_grad_fn():
			self.grad += other.data * variable.grad
			other.grad += self.data * variable.grad
		
		variable._back_grad_fn = _back_grad_fn
		return variable


	def backward(self, grad: Variable | np.ndarray = None) -> None:
		if grad is None:
			grad = np.array([1])
		
		if not isinstance(grad, (Variable, np.ndarray)):
			raise ValueError("The backward gradient must be a numpy array")
		
		if isinstance(grad, Variable):
			grad = grad.grad
		
		self.grad = grad
		variable_queue = [self]

		while len(variable_queue):
			variable = variable_queue.pop(0)
			variable._back_grad_fn()
			variable_queue.extend(list(variable.parents))


if __name__ == "__main__":
	pass