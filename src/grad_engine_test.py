from grad_engine import Variable
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)


def test_add():
    variable_one = Variable(np.array([10, 20]))
    variable_two = Variable(np.array([30, 40]))
    result = variable_one + variable_two
    expected_result = np.array([40, 60])
    if not np.all(result.data == expected_result):
        logging.warning(f"Exptected: {expected_result} and got: {result.data}")
    else:
        logging.info("Passed")


def test_multiply():
    pass



if __name__ == "__main__":
    test_add()
    test_multiply()