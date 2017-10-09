import unittest
import numpy as np
from op_test import OpTest


class SeqPoolType(OpTest):
    AVERAGE = 0
    SUM = 1
    SQRT = 2
    MAX = 3
    LAST = 4
    FIRST = 5


class TestSeqAvgPool(OpTest):
    def set_data(self):
        self.op_type = 'sequence_pool'
        # one level, batch size is 4
        x = np.random.uniform(0.1, 1, [11, 23]).astype('float32')
        lod = [[0, 4, 5, 8, 11]]
        self.inputs = {'X': (x, lod)}

        out = np.zeros((4, 23)).astype('float32')
        self.outputs = {'Out': out}

    def compute(self):
        self.attrs = {'strategy': SeqPoolType.AVERAGE}
        x, lod = self.inputs['X']
        out = self.outputs['Out']
        for i in range(4):
            sub_x = x[lod[0][i]:lod[0][i + 1], :]
            out[i] = sub_x.mean(axis=0)

    def setUp(self):
        self.set_data()
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestSeqAvgPool2D(TestSeqAvgPool):
    def set_data(self):
        self.op_type = 'sequence_pool'
        # one level, batch size is 4
        x = np.random.uniform(0.1, 1, [13, 3, 17]).astype('float32')
        lod = [[0, 4, 5, 8, 13]]
        self.inputs = {'X': (x, lod)}

        out = np.zeros((4, 3, 17)).astype('float32')
        self.outputs = {'Out': out}

    def compute(self):
        self.attrs = {'strategy': SeqPoolType.AVERAGE}
        x, lod = self.inputs['X']
        out = self.outputs['Out']
        for i in range(4):
            sub_x = np.reshape(x[lod[0][i]:lod[0][i + 1], :], (-1, 3 * 17))
            out[i] = np.reshape(sub_x.mean(axis=0), (3, 17))


class TestSeqSumPool(TestSeqAvgPool):
    def compute(self):
        self.attrs = {'strategy': SeqPoolType.SUM}
        x, lod = self.inputs['X']
        out = self.outputs['Out']
        for i in range(4):
            sub_x = x[lod[0][i]:lod[0][i + 1], :]
            out[i] = sub_x.sum(axis=0)


class TestSeqSumPool2D(TestSeqAvgPool2D):
    def compute(self):
        self.attrs = {'strategy': SeqPoolType.SUM}
        x, lod = self.inputs['X']
        out = self.outputs['Out']
        for i in range(4):
            sub_x = np.reshape(x[lod[0][i]:lod[0][i + 1], :], (-1, 3 * 17))
            out[i] = np.reshape(sub_x.sum(axis=0), (3, 17))


if __name__ == '__main__':
    unittest.main()
