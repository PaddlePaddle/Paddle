import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestConcatOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "concat"
        x0 = np.random.random((3, 4)).astype('float32')
        x1 = np.random.random((4, 4)).astype('float32')
        x2 = np.random.random((5, 4)).astype('float32')
        self.inputs = {'X': np.asarray([x0, x1, x2])}
        self.attrs = {'axis': 0}
        self.outpus = {'Out': np.concatenate((x0, x1, x2), axis=0)}


if __name__ == '__main__':
    unittest.main()
