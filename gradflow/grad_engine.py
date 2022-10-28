from __future__ import annotations
from enum import Enum
from typing import Tuple, Union

import numpy as np

from gradflow.data_containers import NumpyDataContainer, DataContainerBase, GPUDataContainer


class Device(str, Enum):
    CPU = "cpu"
    GPU = "gpu"

_device_to_container_map = {
    "cpu": NumpyDataContainer,
    "gpu": GPUDataContainer
}


class Variable:
    def __init__(
        self, 
        data: Variable | np.ndarray | DataContainerBase, 
        parents: Tuple[Variable] = None, 
        requires_grad: bool = False,
        device: str = Device.CPU.value
    ) -> None:
        if isinstance(data, Variable):
            data = data.data
        elif isinstance(data, np.ndarray):
            data = _device_to_container_map[device](data)
        elif isinstance(data, DataContainerBase):
            pass
        else:
            raise TypeError(f"The data type provided is not supported: {type(data)}")

        self.data = data
        self.grad: Union[int, None] = .0 if requires_grad else None
        self.parents = parents or ()
        self._requires_grad = requires_grad
        self._back_grad_fn = lambda: None
        self.device = device


    def __add__(self, other: Variable) -> Variable:
        if not isinstance(other, Variable):
            raise TypeError("The second operator must be a Variable type")
        self._assert_same_device(self, other)

        result = self.data + other.data
        variable = Variable(result, parents=(self, other))

        if any((parent.requires_grad for parent in variable.parents)):
            variable.requires_grad = True

            def _back_grad_fn():
                self._accumulate_gradient(self, variable.grad)
                self._accumulate_gradient(other, variable.grad)

            variable._back_grad_fn = _back_grad_fn
        return variable


    def __sub__(self, other: Variable) -> Variable:
        if not isinstance(other, Variable):
            raise TypeError("The second operator must be a Variable type")
        self._assert_same_device(self, other)

        result = self.data - other.data
        variable = Variable(result, parents=(self, other))

        if any((parent.requires_grad for parent in variable.parents)):
            variable.requires_grad = True

            def _back_grad_fn():
                self._accumulate_gradient(self, variable.grad)
                self._accumulate_gradient(other, -variable.grad)

            variable._back_grad_fn = _back_grad_fn
        return variable


    def __matmul__(self, other: Variable) -> Variable:
        if not isinstance(other, Variable):
            raise TypeError("The second operator must be a Variable type")
        self._assert_same_device(self, other)

        result = self.data @ other.data
        variable = Variable(result, parents=(self, other))

        if any((parent.requires_grad for parent in variable.parents)):
            variable.requires_grad = True

            def _back_grad_fn():
                self._accumulate_gradient(self, other.data * variable.grad)
                self._accumulate_gradient(other, self.data * variable.grad)

            variable._back_grad_fn = _back_grad_fn
        return variable


    def __pow__(self, exponent: int) -> Variable:
        if not isinstance(exponent, int):
            raise TypeError("For power operation the exponent must be a scalar integer value")

        result = self.data ** exponent
        variable = Variable(result, parents=(self), requires_grad=self.requires_grad)

        if variable.requires_grad:
            def _back_grad_fn():
                self._accumulate_gradient(self, exponent * (self.data ** (exponent - 1)) * variable.grad)

            variable._back_grad_fn = _back_grad_fn


    def T(self) -> Variable:
        variable = Variable(self.data.T(), parents=(self,), requires_grad=self.requires_grad)
        
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
        return f"Variable(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"


    @property
    def requires_grad(self) -> bool:
        return self._requires_grad


    @requires_grad.setter
    def requires_grad(self, requires_grad: bool):
        self.grad = .0 if requires_grad else None
        self._requires_grad = requires_grad
    

    @staticmethod
    def _accumulate_gradient(variable: Variable, grad: np.ndarray):
        if variable.requires_grad:
            variable.grad += grad


    def _assert_same_device(self, first_operand: Variable, second_operand: Variable) -> None:
        if first_operand.device != second_operand.device:
            raise Exception("Both the first and the second device of the operation must have the same value.")


    def to(self, device: str) -> Variable:
        if self.device == device:
            return self

        if device == Device.CPU.value:
            return Variable(self.data.item(), device=Device.CPU.value)
        elif device == Device.GPU.value:
            return Variable(self.data.data, device=Device.GPU.value)
        else:
            raise ValueError("Unsupported device.")


if __name__ == "__main__":
    pass
