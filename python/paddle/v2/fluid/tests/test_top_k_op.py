import unittest
import numpy as np
from op_test import OpTest


class TestTopkOp(OpTest):
    def setUp(self):
        self.op_type = "top_k"
        k = 1
        input = np.random.random((32, 84)).astype("float32")
        output = np.ndarray((32, k))
        indices = np.ndarray((32, k)).astype("int64")

        self.inputs = {'X': input}
        self.attrs = {'k': k}

        for rowid in xrange(32):
            row = input[rowid]
            output[rowid] = np.sort(row)[-k:]
            indices[rowid] = row.argsort()[-k:]

        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        self.check_output()


class TestTopkOp3d(OpTest):
    def setUp(self):
        self.op_type = "top_k"
        k = 1
        input = np.random.random((32, 2, 84)).astype("float32")
        input_flat_2d = input.reshape(64, 84)
        output = np.ndarray((64, k))
        indices = np.ndarray((64, k)).astype("int64")

        # FIXME: should use 'X': input for a 3d input
        self.inputs = {'X': input_flat_2d}
        self.attrs = {'k': k}

        for rowid in xrange(64):
            row = input_flat_2d[rowid]
            output[rowid] = np.sort(row)[-k:]
            indices[rowid] = row.argsort()[-k:]

        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
