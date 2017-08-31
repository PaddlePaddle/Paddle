import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestTopkOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "top_k"
        k = 1
        input = np.random.random((32, 84)).astype("float32")
        output = np.ndarray((32, k))
        indices = np.ndarray((32, k))

        self.inputs = {'X': input}
        self.attrs = {'k': k}

        for rowid in xrange(32):
            row = input[rowid]
            output[rowid] = np.sort(row)[-k:]
            indices[rowid] = row.argsort()[-k:]

        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp3d(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "top_k"
        k = 1
        input = np.random.random((32, 2, 84)).astype("float32")
        input_flat_2d = input.reshape(64, 84)
        output = np.ndarray((64, k))
        indices = np.ndarray((64, k))

        # FIXME: should use 'X': input for a 3d input
        self.inputs = {'X': input_flat_2d}
        self.attrs = {'k': k}

        for rowid in xrange(64):
            row = input_flat_2d[rowid]
            output[rowid] = np.sort(row)[-k:]
            indices[rowid] = row.argsort()[-k:]

        self.outputs = {'Out': output, 'Indices': indices}


if __name__ == '__main__':
    unittest.main()
