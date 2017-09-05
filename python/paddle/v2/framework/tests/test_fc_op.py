import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


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
        self.outputs = {"mul_out": add_out, "Y": sigmoid_out}


class TestFCGradOp(GradientChecker):
    def test_normal(self):
        print "nothing"


if __name__ == '__main__':
    unittest.main()
