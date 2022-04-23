from typing import List
from abc import ABC, abstractmethod

import numpy as np

from gradflow.grad_engine import Variable


class BaseOptimizer(ABC):
    def __init__(self, parameters: List[Variable], lr=0.0001) -> None:
        super().__init__()
        self._parameters = parameters
        self._lr = lr


    def zero_grad(self):
        for parameter in self._parameters:
            parameter.grad = np.array([0])


    @abstractmethod
    def step(self):
        raise NotImplementedError


class NaiveSGD(BaseOptimizer):
    def __init__(self, parameters: List[Variable]) -> None:
        super().__init__(parameters=parameters)

    def step(self):
        for parameter in self._parameters:
            delta = np.zeros_like(parameter.data)
            delta += -self._lr * parameter.grad
            parameter.data = parameter.data + delta

