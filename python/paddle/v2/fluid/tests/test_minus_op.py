import unittest
import numpy as np
from op_test import OpTest


class TestMinusOp(OpTest):
    def setUp(self):
        self.op_type = "minus"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((32, 84)).astype("float32")
        }
        self.outputs = {'Out': (self.inputs['X'] - self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


if __name__ == "__main__":
    unittest.main()
