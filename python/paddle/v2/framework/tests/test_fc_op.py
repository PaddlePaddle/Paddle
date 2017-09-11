import unittest
import numpy as np
from op_test import OpTest


class TestFCOp1(OpTest):
    def setUp(self):
        self.op_type = "fc"
        x1 = np.random.random((16, 32)).astype("float32")
        w1 = np.random.random((32, 10)).astype("float32")
        b = np.random.random(10).astype("float32")
        self.inputs = {"X": {"X1": x1}, "W": {"W1": w1}, "b": b}
        mul_out1 = np.dot(x1, w1)
        sum_out = mul_out1
        add_out = sum_out + b
        identity_out = add_out
        self.outputs = {
            "mul_out": {
                "mul_out1": mul_out1,
            },
            "sum_out": sum_out,
            "add_out": add_out,
            "Y": identity_out
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X1", "W1", "b"], "Y", max_relative_error=0.05)


class TestFCOp2(OpTest):
    def setUp(self):
        self.op_type = "fc"
        x1 = np.random.random((16, 32)).astype("float32")
        x2 = np.random.random((16, 32)).astype("float32")
        w1 = np.random.random((32, 10)).astype("float32")
        w2 = np.random.random((32, 10)).astype("float32")
        b = np.random.random(10).astype("float32")
        self.inputs = {
            "X": {
                "X1": x1,
                "X2": x2
            },
            "W": {
                "W1": w1,
                "W2": w2
            },
            "b": b
        }
        #self.attrs = {"activation": "sigmoid"}
        mul_out1 = np.dot(x1, w1)
        mul_out2 = np.dot(x2, w2)
        sum_out = mul_out1 + mul_out2
        add_out = np.add(sum_out, b)
        #sigmoid_out = 1 / (1 + np.exp(-add_out))
        sigmoid_out = add_out
        self.outputs = {
            "mul_out": {
                "mul_out0": mul_out1,
                "mul_out1": mul_out2
            },
            "sum_out": sum_out,
            "add_out": add_out,
            "Y": sigmoid_out
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["X1", "X2", "W1", "W2", "b"], "Y", max_relative_error=0.05)


if __name__ == '__main__':
    unittest.main()
