from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Set
import os

from gradflow.grad_engine import Variable

DEFAULT_MODULE_NAME = "Module"


class Module(ABC):
    def __init__(self, training = True) -> None:
        self._parameters: List[Variable] = []
        self._modules: List[Module] = []
        self._training = training
        self._module_name = DEFAULT_MODULE_NAME


    @abstractmethod
    def forward(self, input: Variable) -> Variable:
        raise NotImplemented


    def add_parameter(self, parameter: Variable) -> Variable:
        self._parameters.append(parameter)
        return parameter


    def add_module(self, module: Module) -> Module:
        self._modules.append(module)
        return module


    def __call__(self, input: Variable) -> Variable:
        return self.forward(input)


    @property
    def modules(self) -> List[Module]:
        return self._modules


    @property
    def parameters(self) -> List[Variable]:
        modules_parameters = []
        modules = [self]
        visited_modules: Set[Module] = set([])

        while len(modules) != 0:
            module = modules.pop()

            if module in visited_modules:
                raise RecursionError("Module already visited, cycle detected.")

            modules_parameters.extend(module._parameters)
            modules.extend(module.modules)
            visited_modules.add(module)

        return modules_parameters


    def module_name(self) -> str:
        return self._module_name


    def __repr__(self) -> str:
        children_modules_description = ""
        if len(self.modules) != 0:
            children_modules_description = "Children modules:"
            modules_description_list = []

            for module in self.modules:
                modules_description_list.append(module.module_name)

            children_modules_description = f"Children modules: {os.linesep} {os.linesep.join(modules_description_list)}"

        return self.module_name + children_modules_description
