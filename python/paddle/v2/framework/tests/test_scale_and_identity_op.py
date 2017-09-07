import unittest
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op
import numpy as np
from paddle.v2.framework.op import Operator


class IdentityTest(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "identity"
        self.inputs = {'X': np.random.random((32, 784)).astype("float32")}
        self.outputs = {'Y': self.inputs['X']}


class IdentityGradOpTest(GradientChecker):
    def test_normal(self):
        op = create_op("identity")
        inputs = {"X": np.random.random((10, 10)).astype("float32")}
        self.check_grad(op, inputs, set("X"), "Y")


class ScaleTest(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "scale"
        self.inputs = {'X': np.random.random((32, 784)).astype("float32")}
        self.attrs = {'scale': -2.3}
        self.outputs = {'Out': self.inputs['X'] * self.attrs['scale']}


class ScaleGradTest(GradientChecker):
    def test_normal(self):
        op = Operator("scale", X="X", Out="Out", scale=3.2)
        self.check_grad(op,
                        {"X": np.random.random((10, 10)).astype("float32")},
                        set("X"), "Out")


if __name__ == '__main__':
    unittest.main()
