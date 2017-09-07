import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestReshapeOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "reshape"
        self.inputs = {'X': np.random.random((2, 4)).astype("float32"), }
        print self.inputs
        self.attrs = {'shape': [4, 2]}
        self.outputs = {'Out': self.inputs['X'].reshape(self.attrs['shape'])}
        print self.outputs


class ReshapeGradOpTest(GradientChecker):
    def test_normal(self):
        op = create_op("reshape")
        inputs = {"X": np.random.random((2, 4)).astype("float32")}
        attrs = {'shape': [4, 2]}
        self.check_grad(op, inputs, attrs, set("X"), "Out")


if __name__ == '__main__':
    unittest.main()
