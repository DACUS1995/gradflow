from __future__ import annotations
from typing import List, Tuple, Dict, Union

import numpy as np


class Variable:
    def __init__(
        self, 
        data: Variable | np.ndarray, 
        parents: Tuple[Variable] = None, 
        requires_grad: bool = False
    ) -> None:
        if isinstance(data, self.__class__):
            data = data.data
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"The data type provided is not supported: {type(data)}")

        self.data = data
        self.grad = 0
        self.parents = parents or ()
        self.requires_grad = requires_grad
        self._back_grad_fn = lambda: None


    def __add__(self, other: Variable) -> Variable:
        if not isinstance(other, Variable):
            raise TypeError("The second operator must be a Variable type")

        result = self.data + other.data
        variable = Variable(result, parents=(self, other))

        if any((parent.requires_grad for parent in variable.parents)):
            variable.requires_grad = True

            def _back_grad_fn():
                self.grad += variable.grad
                other.grad += variable.grad

            variable._back_grad_fn = _back_grad_fn
        return variable


    def __matmul__(self, other: Variable) -> Variable:
        if not isinstance(other, Variable):
            raise TypeError("The second operator must be a Variable type")

        result = np.matmul(self.data, other.data)

        if not isinstance(result, np.ndarray):
            result = np.array([result])

        variable = Variable(result, parents=(self, other))

        if any((parent.requires_grad for parent in variable.parents)):
            variable.requires_grad = True

            def _back_grad_fn():
                self.grad += other.data * variable.grad
                other.grad += self.data * variable.grad

            variable._back_grad_fn = _back_grad_fn
        return variable


    def __pow__(self, exponent: int) -> Variable:
        if not isinstance(exponent, int):
            raise TypeError("For power operation the exponent must be a scalar integer value")


    def T(self) -> Variable:
        variable = Variable(np.transpose(self.data), parents=(self,))
        
        def _back_grad_fn():
            self.grad += variable.grad

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

        # TODO check if topological sort is needed
        while len(variable_queue):
            variable = variable_queue.pop(0)
            variable._back_grad_fn()
            variable_queue.extend(list(variable.parents))


    def __repr__(self):
         return f"Variable(data={self.data}, grad={self.grad})"


if __name__ == "__main__":
    pass