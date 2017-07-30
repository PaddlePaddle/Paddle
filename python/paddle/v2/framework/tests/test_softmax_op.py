import unittest
from op_test_util import OpTestMeta
import numpy as np


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestSoftmaxOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "softmax"
        self.X = np.random.random((32, 100)).astype("float32")
        self.Y = np.apply_along_axis(stable_softmax, 1, self.X)


if __name__ == '__main__':
    unittest.main()
