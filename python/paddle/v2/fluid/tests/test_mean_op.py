import unittest
import numpy as np
from op_test import OpTest


class TestMeanOp(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.inputs = {'X': np.random.random((10, 10)).astype("float32")}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def test_check_output(self):
        self.check_output()

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
