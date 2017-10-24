import unittest
import numpy as np
from op_test import OpTest


class TestMultiplexOp(OpTest):
    def setUp(self):
        self.op_type = "multiplex"
        rows = 4
        index = np.arange(0, rows).astype('int32')
        np.random.shuffle(index)
        index = np.reshape(index, (rows, 1))
        ins1 = np.random.random((rows, 10)).astype("float32")
        ins2 = np.random.random((rows, 10)).astype("float32")
        ins3 = np.random.random((rows, 10)).astype("float32")
        ins4 = np.random.random((rows, 10)).astype("float32")
        self.inputs = {
            'Ids': index,
            'X': [('x1', ins1), ('x2', ins2), ('x3', ins3), ('x4', ins4)]
        }
        # multiplex output
        output = np.zeros_like(ins1)
        for i in range(0, rows):
            k = index[i][0]
            output[i] = self.inputs['X'][k][1][i]
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x1', 'x2', 'x3', 'x4'], 'Out')

    def test_check_grad_ignore_x1(self):
        self.check_grad(['x2', 'x3', 'x4'], 'Out', no_grad_set=set('x1'))

    def test_check_grad_ignore_x1_x2(self):
        self.check_grad(['x3', 'x4'], 'Out', no_grad_set=set(['x1', 'x2']))

    def test_check_grad_ignore_x3(self):
        self.check_grad(['x1', 'x2', 'x4'], 'Out', no_grad_set=set('x3'))


if __name__ == '__main__':
    unittest.main()
