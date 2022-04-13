from typing import List
from gradflow.grad_engine import Variable

class Module:
    def __init__(self) -> None:
        self._parameters: List[Variable] = []
        self._modules: List[Module] = []

    def forward(self, input: Variable) -> Variable:
        raise NotImplemented()

    def add_parameter(self, parameter: Variable) -> None:
        self._parameters.append(parameter)

    def add_module(self, module: Module) -> None:
        self._modules.append(module)

    @property
    def modules(self):
        return self._modules

    @property
    def parameters(self):
        return self._parameters