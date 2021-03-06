from turtle import forward
from gradflow.grad_engine import Variable
import numpy as np

from gradflow.layers.module import Module

class Linear(Module):
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        weights_data: np.ndarray = np.random.uniform(size=in_size * out_size).reshape((in_size, out_size))
        self.weights = Variable(weights_data, requires_grad=True)
        self.b = Variable(np.random.uniform(size=out_size), requires_grad=True)

        self.add_parameter(self.weights)
        self.add_parameter(self.b)

    def forward(self, input: Variable):
        tmp = input @ self.weights
        out = tmp + self.b
        return out
