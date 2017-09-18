import unittest
import numpy as np
from op_test import OpTest


class TestSeqAvgPool1D(OpTest):
    def setUp(self):
        self.op_type = 'sequence_avg_pool'
        # one level, batch size is 4
        x = np.random.uniform(0.1, 1, [11, 23]).astype('float32')
        lod = [[0, 4, 5, 8, 11]]

        out = np.zeros((4, 23)).astype('float32')
        for i in range(4):
            sub_x = x[lod[0][i]:lod[0][i + 1], :]
            out[i] = sub_x.mean(axis=0)

        self.inputs = {'X': (x, lod)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestSeqAvgPool2D(OpTest):
    def setUp(self):
        self.op_type = 'sequence_avg_pool'
        # one level, batch size is 4
        x = np.random.uniform(0.1, 1, [13, 3, 17]).astype('float32')
        lod = [[0, 4, 5, 8, 13]]

        out = np.zeros((4, 3, 17)).astype('float32')
        for i in range(4):
            sub_x = np.reshape(x[lod[0][i]:lod[0][i + 1], :], (-1, 3 * 17))
            out[i] = np.reshape(sub_x.mean(axis=0), (3, 17))

        self.inputs = {'X': (x, lod)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


if __name__ == '__main__':
    unittest.main()
