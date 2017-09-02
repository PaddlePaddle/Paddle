import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestSumOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "sum"
        x0 = np.random.random((3, 4)).astype('float32')
        x1 = np.random.random((3, 4)).astype('float32')
        x2 = np.random.random((3, 4)).astype('float32')
        y = x0 + x1 + x2
        self.inputs = {'X': [x0, x1, x2]}
        self.outputs = {'Out': y}


if __name__ == '__main__':
    unittest.main()
