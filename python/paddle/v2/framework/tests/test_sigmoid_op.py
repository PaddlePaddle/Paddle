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
    def test_compare_grad(self):
        op = create_op("sigmoid")
        inputs = {"X": np.random.random((11, 17)).astype("float32")}

        # compare gpu and cpu results for backward op
        self.compare_grad(op, inputs)

    def test_check_grad(self):
        op = create_op("sigmoid")
        inputs = {"X": np.random.random((11, 17)).astype("float32")}

        # check gradients
        self.check_grad(op, inputs, set("X"), "Y")


if __name__ == '__main__':
    unittest.main()
