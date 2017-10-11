import unittest
import numpy as np
from op_test import OpTest


class TestInterpOp(OpTest):
    def setUp(self):
        self.op_type = "interp"
        x = np.random.random((2, 3)).astype("float32")
        y = np.random.random((2, 3)).astype("float32")
        w = np.random.random(2).astype("float32")

        sub_out = x - y
        mul_out = sub_out * w.reshape(2, 1)
        out = mul_out + y

        self.inputs = {'X': x, 'Y': y, 'W': w}
        self.outputs = {'Out': out, 'SubOut': sub_out, 'MulOut': mul_out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


if __name__ == "__main__":
    unittest.main()
