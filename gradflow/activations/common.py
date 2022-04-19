import numpy as np

from gradflow.grad_engine import Variable

def relu(input: Variable) -> Variable:
    result = np.maximum(input.data, 0)
    variable = Variable(result, parents=(input,))

    if input.requires_grad:
        def _back_grad_fn():
            input.grad += (variable.data > 0) * variable.grad
        variable._back_grad_fn = _back_grad_fn
        variable.requires_grad = True

    return variable


def softmax(input: Variable) -> Variable:
    pass