import unittest
import numpy as np
from op_test import OpTest


class TestConcatOp(OpTest):
    def setUp(self):
        self.op_type = "concat"
        x0 = np.random.random((2, 1, 4, 5)).astype('float32')
        x1 = np.random.random((2, 2, 4, 5)).astype('float32')
        x2 = np.random.random((2, 3, 4, 5)).astype('float32')
        axis = 1
        self.inputs = {'X': [('x0', x0), ('x1', x1), ('x2', x2)]}
        self.attrs = {'axis': axis}
        self.outputs = {'Out': np.concatenate((x0, x1, x2), axis=axis)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')


if __name__ == '__main__':
    unittest.main()
