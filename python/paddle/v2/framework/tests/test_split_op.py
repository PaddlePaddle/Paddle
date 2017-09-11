import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestConcatOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "split"
        axis = 0
        indices = 1
        axis = 0
        x = np.random.random((3, 2)).astype('float32')
        self.inputs = {'X': x}
        #self.attrs = {'axis': axis, 'indices': indices}
        #self.outputs = {'Out': np.split(x, indices, axis)}
        self.outputs = {'Out': x}


if __name__ == '__main__':
    unittest.main()
