import unittest
import numpy as np
from op_test import OpTest


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestSoftmaxOp(OpTest):
    def setUp(self):
        self.op_type = "softmax"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [10, 10]).astype("float32")
        }
        self.outputs = {
            'Y': np.apply_along_axis(stable_softmax, 1, self.inputs['X'])
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y')


if __name__ == "__main__":
    unittest.main()
