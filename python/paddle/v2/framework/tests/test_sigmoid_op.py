import unittest
import numpy as np
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op


class TestSigmoidOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "sigmoid"
        self.inputs = {'X': np.random.random((15, 31)).astype("float32")}
        self.outputs = {'Y': 1 / (1 + np.exp(-self.inputs['X']))}


class TestSigmoidGradOp(GradientChecker):
    def test_grad(self):
        op = create_op("sigmoid")
        inputs = {"X": np.random.uniform(0.1, 1, [11, 17]).astype("float32")}
        # compare gpu and cpu results for backward op.
        # this test will be skiped if only compiling CPU version.
        self.compare_grad(op, inputs)
        # check gradients 
        self.check_grad(op, inputs, set("X"), "Y", max_relative_error=0.007)


if __name__ == '__main__':
    unittest.main()
