import unittest
import numpy as np
from op_test import OpTest


class TestFillConstantBatchSizeLikeWhenFirstDimIsBatchSize(OpTest):
    def setUp(self):
        self.op_type = "fill_constant_batch_size_like"
        self.inputs = {'Input': np.random.random((219, 232)).astype("float32")}
        self.attrs = {'value': 3.5, 'shape': [-1, 132, 7]}

        out = np.random.random((219, 132, 7)).astype("float32")
        out.fill(3.5)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


class TestFillConstantBatchSizeLikeWhenSecondDimIsBatchSize(OpTest):
    def setUp(self):
        self.op_type = "fill_constant_batch_size_like"
        self.inputs = {'Input': np.random.random((219, 232)).astype("float32")}
        self.attrs = {
            'value': 3.5,
            'shape': [132, -1, 7],
            'input_dim_idx': 0,
            'output_dim_idx': 1
        }

        out = np.random.random((132, 219, 7)).astype("float32")
        out.fill(3.5)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
