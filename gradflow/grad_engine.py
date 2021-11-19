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


	def __add__(self, other: Variable) -> Variable:
		result =  Variable(self.data + other.data)

		def _back_grad_fn():
			self.grad += result.grad
			other.grad += result.grad

		result._back_grad_fn = _back_grad_fn
		return Variable(result)


	def __mul__(self, other: Variable):
		self.grad = other.data
		other.grad = self.data
		result = Variable(np.matmul(self.data, other.data))

		def _back_grad_fn():
			self.grad += other.data * result.grad
			other.grad += self.data * result.grad
		
		result._back_grad_fn = _back_grad_fn
		return Variable(result)


	def backward(self) -> None:
		pass


if __name__ == "__main__":
	pass