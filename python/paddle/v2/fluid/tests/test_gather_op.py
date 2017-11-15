import unittest
import numpy as np
from op_test import OpTest


class TestGatherOp(OpTest):
    def setUp(self):
        self.op_type = "gather"
        xnp = np.random.random((10, 20)).astype("float32")
        self.inputs = {'X': xnp, 'Index': np.array([1, 3, 5]).astype("int32")}
        self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
