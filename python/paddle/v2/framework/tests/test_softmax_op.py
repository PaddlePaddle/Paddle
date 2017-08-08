import unittest

import numpy as np

from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestSoftmaxOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "softmax"
        self.inputs = {'X': np.random.random((32, 100)).astype("float32")}
        self.outputs = {
            'Y': np.apply_along_axis(stable_softmax, 1, self.inputs['X'])
        }


class SoftmaxGradOpTest(GradientChecker):
    def test_softmax(self):
        op = create_op("softmax")
        inputs = {"X": np.random.uniform(0.1, 1, [10, 10]).astype("float32")}
        self.check_grad(op, inputs, set("X"), "Y")


if __name__ == '__main__':
    unittest.main()
