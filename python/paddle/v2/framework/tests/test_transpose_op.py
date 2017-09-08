import unittest
import numpy as np
from gradient_checker import GradientChecker
from op_test_util import OpTestMeta
from paddle.v2.framework.op import Operator


class TestTransposeOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "transpose"
        self.inputs = {'X': np.random.random((3, 4)).astype("float32"), }
        self.attrs = {'axis': [1, 0]}
        self.outputs = {'Out': self.inputs['X'].transpose((1, 0))}


class TransposeGradOpTest(GradientChecker):
    def test_transpose(self):
        op = Operator("transpose", X="X", Out="Out", axis=[1, 0])
        inputs = {'X': np.random.random((32, 84)).astype("float32"), }

        self.check_grad(op, inputs, set(["X"]), "Out", max_relative_error=0.5)


if __name__ == '__main__':
    unittest.main()
