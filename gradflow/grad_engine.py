from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Union


class Variable:
	def __init__(self, data: Variable | np.ndarray, parents: Tuple = ()) -> None:
		if isinstance(data, self.__class__):
			data = data.data
		elif not isinstance(data, np.ndarray):
			raise TypeError("The data type provided is not supported")

		self.data = data
		self.grad = 0
		self.parents = parents
		self._back_grad_fn = lambda x: None


	def __add__(self, other: Variable) -> Variable:
		result =  Variable(self.data + other.data)

		def _back_grad_fn():
			self.grad += result.grad
			other.grad += result.grad

		variable = Variable(result, parents=(self, other))
		variable._back_grad_fn = _back_grad_fn
		return variable

	def __mul__(self, other: Variable):
		self.grad = other.data
		other.grad = self.data
		result = Variable(np.matmul(self.data, other.data))

		def _back_grad_fn():
			self.grad += other.data * result.grad
			other.grad += self.data * result.grad
		
		variable = Variable(result, parents=(self, other))
		variable._back_grad_fn = _back_grad_fn
		return variable


	def backward(self) -> None:
		variables = [self]
		while len(variables):
			variable = variables.pop(0)
			variable._back_grad_fn()
			variables.extend(list(self.parents))


if __name__ == "__main__":
	pass