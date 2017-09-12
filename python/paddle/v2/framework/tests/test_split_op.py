import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestConcatOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "split"
        axis = 0
        indices = 2
        split_info = []
        x = np.random.random((4, 2)).astype('float32')
        assert x.shape(axis) % indices == 0
        split_info = [x.shape(axis) / indices for i in xrange(n)]
        self.inputs = {'X': x}
        self.attrs = {'axis': axis, 'split_info': split_info}
        print np.split(x, indices, axis)
        self.outputs = {'Out': list(np.split(x, indices, axis))}


if __name__ == '__main__':
    unittest.main()
