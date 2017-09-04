import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestRowL2NormOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "row_l2_norm"
        self.inputs = {'X': np.random.random((10, 10)).astype("float32")}
        eps = 1e-6
        l2_norm = np.sqrt(np.square(self.inputs['X']).sum(axis=1))[:,
                                                                   np.newaxis]
        out = self.inputs['X'] / ((l2_norm + eps))
        self.outputs = {'L2_Norm': l2_norm, 'Out': out}


class TestRowL2NormGradOp(GradientChecker):
    def test_row_l2_norm(self):
        op = create_op("row_l2_norm")
        # Since error will be high when grad nearly 0, use small size and
        # big threshold to mitigate.
        inputs = {'X': np.random.random((3, 10)).astype("float32")}
        self.check_grad(op, inputs, set(["X"]), "Out", max_relative_error=0.05)


if __name__ == '__main__':
    unittest.main()
