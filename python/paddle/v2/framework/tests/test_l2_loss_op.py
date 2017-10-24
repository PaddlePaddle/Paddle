import numpy as np
import unittest
from numpy import linalg as LA
from op_test import OpTest


class TestL2LossOp(OpTest):
    """Test l2_loss
    """

    def setUp(self):
        self.op_type = "l2_loss"
        self.max_relative_error = 0.05

        X = np.random.uniform(-1, 1, (13, 19)).astype("float32")
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {'X': X}
        self.outputs = {'Out': np.square(LA.norm(X)) * 0.5}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=self.max_relative_error)


if __name__ == "__main__":
    unittest.main()
