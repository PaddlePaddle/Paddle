import unittest
import numpy as np
from op_test import OpTest


class TestSignOp(OpTest):
    def setUp(self):
        self.op_type = "sign"
        self.inputs = {
            'X': np.random.uniform(-10, 10, (10, 10)).astype("float32")
        }
        self.outputs = {'Out': np.sign(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
