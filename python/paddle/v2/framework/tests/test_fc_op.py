import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta
from paddle.v2.framework.op import Operator


class TestFCOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "fc"
        self.inputs = {
            "X": np.random.random((32, 784)).astype("float32"),
            "W": np.random.random((784, 1000)).astype("float32"),
            "b": np.random.random(1000).astype("float32")
        }
        self.attrs = {"activation": "sigmoid"}
        mul_out = np.dot(self.inputs["X"], self.inputs["W"])
        add_out = np.add(mul_out, self.inputs["b"])
        sigmoid_out = 1 / (1 + np.exp(-add_out))
        self.outputs = {
            "mul_out": mul_out,
            "add_out": add_out,
            "Out": sigmoid_out
        }


class TestFCGradOp(GradientChecker):
    def test_normal(self):
        self.inputs = {
            "X": np.random.random((4, 4)).astype("float32"),
            "W": np.random.random((4, 4)).astype("float32"),
            "b": np.random.random(4).astype("float32")
        }
        op = Operator(
            "fc", X="X", W="W", b="b", Out="Out", activation="sigmoid")
        #self.check_grad(op, self.inputs, ["X", "W", "b"], "Out")


if __name__ == '__main__':
    unittest.main()
