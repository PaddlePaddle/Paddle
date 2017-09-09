import unittest
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op
import numpy as np
from paddle.v2.framework.op import Operator


class TestScalingOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "scaling"
        self.inputs = {
            'X': np.random.random((32, 64)).astype("float32"),
            'weight': np.random.random(32).astype("float32")
        }
        self.outputs = {
            'Out': np.dot(np.diag(self.inputs['weight']), self.inputs['X'])
        }


class ScalingGradOp(GradientChecker):
    def test_scaling(self):
        op = create_op("scaling")
        inputs = {
            'X': np.random.random((32, 64)).astype("float32"),
            'weight': np.random.random(32).astype("float32")
        }
        self.check_grad(
            op, inputs, set(['X', "weight"]), "Out", max_relative_error=0.5)


if __name__ == '__main__':
    unittest.main()
if __name__ == '__main__':
    unittest.main()
