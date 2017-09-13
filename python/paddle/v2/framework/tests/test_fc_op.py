import unittest
import numpy as np
from op_test import OpTest


class TestFCOp1(OpTest):
    def setUp(self):
        self.op_type = "fc"
        x0 = np.random.random((16, 32)).astype("float32")
        w0 = np.random.random((32, 10)).astype("float32")
        b = np.random.random(10).astype("float32")
        self.inputs = {"X": [("X0", x0)], "W": [("W0", w0)], "b": b}
        mul_out0 = np.dot(x0, w0)
        sum_out = mul_out0
        add_out = sum_out + b
        identity_out = add_out
        self.outputs = {
            "mul_out": [("mul_out0", mul_out0)],
            "sum_out": sum_out,
            "add_out": add_out,
            "Y": identity_out
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X0", "W0", "b"], "Y", max_relative_error=0.01)


class TestFCOp2(OpTest):
    def setUp(self):
        self.op_type = "fc"
        x0 = np.random.random((16, 32)).astype("float32")
        x1 = np.random.random((16, 32)).astype("float32")
        w0 = np.random.random((32, 10)).astype("float32")
        w1 = np.random.random((32, 10)).astype("float32")
        b = np.random.random(10).astype("float32")
        self.inputs = {
            "X": [("X0", x0), ("X1", x1)],
            "W": [("W0", w0), ("W1", w1)],
            "b": b
        }
        self.attrs = {"activation": "sigmoid"}
        mul_out0 = np.dot(x0, w0)
        mul_out1 = np.dot(x1, w1)
        sum_out = mul_out0 + mul_out1
        add_out = np.add(sum_out, b)
        sigmoid_out = 1 / (1 + np.exp(-add_out))
        self.outputs = {
            "mul_out": [("mul_out0", mul_out0), ("mul_out1", mul_out1)],
            "sum_out": sum_out,
            "add_out": add_out,
            "Y": sigmoid_out
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["X0", "X1", "W0", "W1", "b"], "Y", max_relative_error=0.01)


if __name__ == '__main__':
    unittest.main()
