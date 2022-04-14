from typing import List, Any

from gradflow.grad_engine import Variable

class Module:
    def __init__(self, training = True) -> None:
        self._parameters: List[Variable] = []
        self._modules: List[Module] = []
        self._training = training

    def forward(self, input: Variable) -> Variable:
        raise NotImplemented()

    def add_parameter(self, parameter: Variable) -> None:
        self._parameters.append(parameter)

    def add_module(self, module: Module) -> None:
        self._modules.append(module)

    def __call__(self, input: Variable) -> Variable:
        return self.forward(input)

    @property
    def modules(self):
        return self._modules

    @property
    def parameters(self):
        return self._parameters