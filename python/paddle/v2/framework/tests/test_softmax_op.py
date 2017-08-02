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


class TestSoftmaxGradOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        batch_size = 10
        class_num = 5
        # n = number of examples, D = |labels|
        # Initialize X and add 1e-2 for numerical stability
        Y = np.random.rand(batch_size, class_num).astype(np.float32)
        Y = Y + 1e-2
        dY = np.random.rand(batch_size, class_num).astype(np.float32)

        # Reference implementation of cross entropy with soft labels
        def label_softmax_grad(Y, dY):
            dX = Y * 0.0
            for i in range(batch_size):
                d = np.dot(Y[i, :], dY[i, :])
                dX[i, :] = Y[i, :] * (dY[i, :] - d)
            return [dX]

        self.type = "softmax_grad"
        self.X = np.random.random((32, 100)).astype("float32")
        self.Y = np.apply_along_axis(stable_softmax, 1, self.X)


if __name__ == '__main__':
    unittest.main()
