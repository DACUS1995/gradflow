from difflib import restore
import numpy as np

from gradflow.grad_engine import Variable
from gradflow.data_containers import NumpyDataContainer

def relu(input: Variable) -> Variable:
    if not isinstance(input.data, NumpyDataContainer):
        raise ValueError("Only supported data container is numy based curently.")

    result = np.maximum(input.data.data, 0)
    variable = Variable(result, parents=(input,))

    if input.requires_grad:
        def _back_grad_fn():
            input.grad += np.transpose((variable.data.data > 0)) * variable.grad
        variable._back_grad_fn = _back_grad_fn
        variable.requires_grad = True

    return variable


def softmax(input: Variable) -> Variable:
	# TODO Abstract the usage of numpy
    # e_x = np.exp(input.data - np.max(input.data))
    e_x = (input.data - input.data.max()).exp()
    result = e_x.data / e_x.data.sum(axis=0)
    return result

    # SM = self.value.reshape((-1,1))
    # jac = np.diagflat(self.value) - np.dot(SM, SM.T)
