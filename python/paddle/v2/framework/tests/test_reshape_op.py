import unittest
import numpy as np
from op_test import OpTest


class TestReshapeOp(OpTest):
    def setUp(self):
        self.op_type = "reshape"
        self.inputs = {'X': np.random.random((10, 20)).astype("float32")}
        self.attrs = {'shape': [10 * 20]}
        self.outputs = {'Out': self.inputs['X'].reshape(self.attrs['shape'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


if __name__ == '__main__':
    unittest.main()
