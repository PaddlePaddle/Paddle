import unittest
import numpy as np
from op_test import OpTest


class TestIdentityOp(OpTest):
    def setUp(self):
        self.op_type = "identity"
        self.inputs = {'X': np.random.random((10, 10)).astype("float32")}
        self.outputs = {'Y': self.inputs['X']}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y')


if __name__ == "__main__":
    unittest.main()
