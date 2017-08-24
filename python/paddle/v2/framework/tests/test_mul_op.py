import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestMulOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "mul"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((84, 100)).astype("float32")
        }
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}


class MulGradOpTest(GradientChecker):
    def test_mul(self):
        op = create_op("mul")
        inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((84, 100)).astype("float32")
        }
        # mul op will enlarge the relative error
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.5)


# TODO(dzh,qijun) : mulgrad test case need transpose feature of blas library

if __name__ == '__main__':
    unittest.main()
