import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestConcatOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "concat"
        x0 = np.random.random((2, 3, 2, 5)).astype('float32')
        x1 = np.random.random((2, 3, 3, 5)).astype('float32')
        x2 = np.random.random((2, 3, 4, 5)).astype('float32')
        axis = 2
        self.inputs = {'X': [x0, x1, x2]}
        self.attrs = {'axis': axis}
        self.outputs = {'Out': np.concatenate((x0, x1, x2), axis=axis)}


if __name__ == '__main__':
    unittest.main()
