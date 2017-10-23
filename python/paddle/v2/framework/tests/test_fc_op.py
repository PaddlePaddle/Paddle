import unittest
import numpy as np
from op_test import OpTest


class TestFCOp1(OpTest):
    def setUp(self):
        x0 = np.random.random((16, 32)).astype("float32")
        w0 = np.random.random((32, 10)).astype("float32")

        mul_out0 = np.dot(x0, w0)
        identity_out = mul_out0

        self.op_type = "fc"
        self.inputs = {"X": [("X0", x0)], "W": [("W0", w0)]}
        self.outputs = {"MulOut": [("MulOut0", mul_out0)], "Out": identity_out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X0", "W0"], "Out", max_relative_error=0.01)


class TestFCOp2(OpTest):
    def setUp(self):
        x0 = np.random.random((16, 4, 8)).astype("float32")
        x1 = np.random.random((4, 4, 32)).astype("float32")
        w0 = np.random.random((32, 10)).astype("float32")
        w1 = np.random.random((32, 10)).astype("float32")
        b = np.random.random(10).astype("float32")

        mul_out0 = np.dot(x0.reshape(16, 4 * 8), w0)
        mul_out1 = np.dot(x1.reshape(4 * 4, 32), w1)
        sum_out = mul_out0 + mul_out1
        add_out = np.add(sum_out, b)
        sigmoid_out = 1 / (1 + np.exp(-add_out))

        self.op_type = "fc"
        self.inputs = {
            "X": [("X0", x0), ("X1", x1)],
            "W": [("W0", w0), ("W1", w1)],
            "B": b
        }
        self.attrs = {"xNumColDims": [1, 2], "activation": "sigmoid"}
        self.outputs = {
            "MulOut": [("MulOut0", mul_out0), ("MulOut1", mul_out1)],
            "SumOut": sum_out,
            "AddOut": add_out,
            "Out": sigmoid_out
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["X0", "X1", "W0", "W1", "B"], "Out", max_relative_error=0.01)


if __name__ == '__main__':
    unittest.main()
