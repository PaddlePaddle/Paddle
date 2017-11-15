import unittest
import numpy as np
from op_test import OpTest


class TestSplitOp(OpTest):
    def setUp(self):
        self.op_type = "split"
        axis = 0
        x = np.random.random((4, 2, 5)).astype('float32')
        out = np.split(x, [1, 3], axis)
        self.inputs = {'X': x}
        self.attrs = {'axis': axis, 'sections': [1, 2, 1]}
        self.outputs = {'Out': [('out%d' % i, out[i]) \
            for i in xrange(len(out))]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'])


if __name__ == '__main__':
    unittest.main()
