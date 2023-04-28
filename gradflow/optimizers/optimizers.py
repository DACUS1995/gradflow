from typing import List
from abc import ABC, abstractmethod

import numpy as np

from gradflow.grad_engine import Variable
from gradflow.utils import no_grad


MIN_GRAD_CLIP = -1000
MAX_GRAD_CLIP = 1000
EPS = 1e-8


class BaseOptimizer(ABC):
    def __init__(self, parameters: List[Variable], lr=0.0001) -> None:
        super().__init__()
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        self._parameters = [param for param in parameters if param._requires_grad]
        self._lr = lr

    def zero_grad(self):
        for parameter in self._parameters:
            if parameter.requires_grad == False:
                continue

            if isinstance(parameter.grad, np.ndarray):
                parameter.grad = np.zeros_like(parameter.grad)
            else:
                parameter.grad = np.array([0], dtype=float)

    @abstractmethod
    def step(self):
        raise NotImplementedError


class NaiveSGD(BaseOptimizer):
    def __init__(self, parameters: List[Variable], lr=0.001) -> None:
        super().__init__(parameters=parameters, lr=lr)

    def step(self):
        with no_grad():
            for parameter in self._parameters:
                clipped_grad = np.clip(parameter.grad, MIN_GRAD_CLIP, MAX_GRAD_CLIP)
                delta = -self._lr * clipped_grad
                delta = np.transpose(delta)
                parameter.data = parameter.data + delta


class Adam(BaseOptimizer):
    def __init__(
            self, 
            parameters: List[Variable], 
            lr: float = 0.0001, 
            beta1: float = 0.9,  
            beta2: float = 0.999,
        ) -> None:
        """_summary_

        Args:
            parameters (List[Variable]): parameters to optimise
            lr (float, optional): learning rate. Defaults to 0.0001.
            beta1 (float, optional): decay rate for first momentum. Defaults to 0.9.
            beta2 (float, optional): decay rate for second momentum. Defaults to 0.999.
        """
        super().__init__(parameters=parameters, lr=lr)
        
        self._beta1 = beta1
        self._beta2 = beta2
        self.mt = []
        self.vt = []
        self.t = 0

        for parameter in self._parameters:
            self.vt.append(np.zeros_like(parameter.data.data))
            self.mt.append(np.zeros_like(parameter.data.data))        
    

    def step(self):
        self.t += 1
        with no_grad():
            for idx, parameter in enumerate(self._parameters):
                clipped_grad = np.clip(parameter.grad, MIN_GRAD_CLIP, MAX_GRAD_CLIP)
                self.mt[idx] = self._beta1 * self.mt[idx] + (1 - self._beta1) * clipped_grad
                self.vt[idx] = self._beta2 * self.vt[idx] + (1 - self._beta2) * (clipped_grad ** 2)

                # Bias correction
                mt_hat = self.mt[idx] / (1 - self._beta1 ** self.t)
                vt_hat = self.vt[idx] / (1 - self._beta2 ** self.t)

                delta = -self._lr * (mt_hat / (vt_hat ** 0.5 + EPS)) 
                parameter.data = parameter.data + delta

