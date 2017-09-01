import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class MinusOpTest(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "minus"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((32, 84)).astype("float32")
        }
        self.outputs = {'Out': (self.inputs['X'] - self.inputs['Y'])}


class MinusGradTest(GradientChecker):
    def test_left(self):
        op = create_op("minus")
        inputs = {
            "X": np.random.random((10, 10)).astype("float32"),
            "Y": np.random.random((10, 10)).astype("float32")
        }
        self.check_grad(op, inputs, ["X", 'Y'], "Out")


if __name__ == '__main__':
    unittest.main()
