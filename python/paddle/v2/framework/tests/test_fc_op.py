import unittest
import numpy as np
from op_test import OpTest
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


class TestFCOp(OpTest):
    def setUp(self):
        print "Run"
        self.op_type = "fc"
        x0 = np.random.random((32, 256)).astype("float32")
        x1 = np.random.random((32, 256)).astype("float32")
        w0 = np.random.random((256, 100)).astype("float32")
        w1 = np.random.random((256, 100)).astype("float32")
        b = np.random.random(100).astype("float32")
        self.inputs = {
            "X": {
                "X0": x0,
                "X1": x1
            },
            "W": {
                "W0": w0,
                "W1": w1
            },
            "b": b
        }
        #self.attrs = {"activation": "sigmoid"}
        mul_out = np.dot(x0, w0) + np.dot(x1, w1)
        add_out = np.add(mul_out, b)
        #sigmoid_out = 1 / (1 + np.exp(-add_out))
        sigmoid_out = add_out
        self.outputs = {
            "mul_out": mul_out,
            "add_out": add_out,
            "Y": sigmoid_out
        }

    def test_check_output(self):
        self.check_output(core.CPUPlace())
        self.check_output(core.GPUPlace(0))

    #def test_check_grad(self):
    #    self.check_grad(["X0", "X1", "W0", "W1", "b"], "Y")


if __name__ == '__main__':
    unittest.main()
