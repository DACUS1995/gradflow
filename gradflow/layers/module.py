from __future__ import annotations
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

    def zero_grad(self):
        raise NotImplementedError()

    def __call__(self, input: Variable) -> Variable:
        return self.forward(input)

    @property
    def modules(self) -> List[Module]:
        return self._modules

    @property
    def parameters(self):
        modules_parameters = []
        modules = [self]
        visited_modules = set(modules)

        while len(modules) != 0:
            module = modules.pop()

            if module in visited_modules:
                raise RecursionError("Module already visited, cycle detected.")

            modules_parameters.extend(module.parameters)
            modules.extend(module.modules)
            visited_modules.add(module)

        return modules_parameters

    def __repr__(self) -> str:
        return f"Children modules: []"
