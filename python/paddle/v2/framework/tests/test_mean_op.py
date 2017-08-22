import unittest
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op
import numpy as np


class TestMeanOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "mean"
        self.inputs = {'X': np.random.random((32, 784)).astype("float32")}
        self.outputs = {'Out': np.mean(self.inputs['X'])}


class MeanGradOpTest(GradientChecker):
    def test_normal(self):
        op = create_op("mean")
        inputs = {"X": np.random.random((10, 10)).astype("float32")}
        self.check_grad(op, inputs, set("X"), "Out")


if __name__ == '__main__':
    unittest.main()
