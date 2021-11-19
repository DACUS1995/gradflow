from gradflow.grad_engine import Variable
import numpy as np
import logging

log = logging.getLogger()


def test_add():
    variable_one = Variable(np.array([10, 20]))
    variable_two = Variable(np.array([30, 40]))
    result = variable_one + variable_two
    expected_result = np.array([40, 60])
    assert np.all(result.data == expected_result)


def test_multiply():
    pass



if __name__ == "__main__":
    test_add()
    test_multiply()