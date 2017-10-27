import numpy as np
import unittest
from op_test import OpTest


class TestL1NormOp(OpTest):
    """Test l1_norm
    """

    def setUp(self):
        self.op_type = "l1_norm"
        self.max_relative_error = 0.005

        X = np.random.uniform(-1, 1, (13, 19)).astype("float32")
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {'X': X}
        self.outputs = {'Out': np.sum(np.abs(X))}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=self.max_relative_error)


if __name__ == "__main__":
    unittest.main()
