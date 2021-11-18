from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Union


class Variable:
	def __init__(self, data: Variable | np.ndarray) -> None:
		if isinstance(data, self.__class__):
			data = data.data
		elif not isinstance(data, np.ndarray):
			raise TypeError("The data type provided is not supported")
		
		self.data = data
		self.grad = None


	def __add__(self, other: Variable) -> Variable:
		result = self.data + other.data
		return Variable(result)


	def __mul__(self, other):
		result = np.matmul(self.data, other.data)
		return Variable(result)


	def backpropagate(self) -> None:
		pass


if __name__ == "__main__":
	pass