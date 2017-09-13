import unittest
import numpy as np
from op_test import OpTest


class TestRelu(OpTest):
    def setUp(self):
        self.op_type = "relu"
        self.inputs = {
            'X': np.random.uniform(-1, 1, [11, 17]).astype("float32")
        }
        self.outputs = {'Y': np.maximum(self.inputs['X'], 0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y", max_relative_error=0.007)


if __name__ == '__main__':
    unittest.main()
