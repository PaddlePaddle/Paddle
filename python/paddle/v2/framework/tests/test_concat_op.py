import unittest
import numpy as np
from op_test import OpTest


class TestConcatOp(OpTest):
    def setUp(self):
        self.op_type = "concat"
        x0 = np.random.random((2, 3, 2, 5)).astype('float32')
        x1 = np.random.random((2, 3, 3, 5)).astype('float32')
        x2 = np.random.random((2, 3, 4, 5)).astype('float32')
        axis = 2
        self.inputs = {'X': [('x0', x0), ('x1', x1), ('x2', x2)]}
        self.attrs = {'axis': axis}
        self.outputs = {'Out': np.concatenate((x0, x1, x2), axis=axis)}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
