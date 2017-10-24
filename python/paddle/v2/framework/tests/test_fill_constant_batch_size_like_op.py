import unittest
import numpy as np
from op_test import OpTest


class TestFillZerosLikeOp(OpTest):
    def setUp(self):
        self.op_type = "fill_constant_batch_size_like"
        self.inputs = {'Input': np.random.random((219, 232)).astype("float32")}
        self.attrs = {'value': 3.5, 'shape': [-1, 132, 777]}

        out = np.random.random((219, 132, 777)).astype("float32")
        out.fill(3.5)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
