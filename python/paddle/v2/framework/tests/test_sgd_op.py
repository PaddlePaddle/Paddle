import unittest
import numpy as np
from op_test import OpTest


class TestSGDOp(OpTest):
    def setUp(self):
        self.op_type = "sgd"
        w = np.random.random((102, 105)).astype("float32")
        g = np.random.random((102, 105)).astype("float32")
        lr = 0.1

        self.inputs = {'param': w, 'grad': g}
        self.attrs = {'learning_rate': lr}
        self.outputs = {'param_out': w - lr * g}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
