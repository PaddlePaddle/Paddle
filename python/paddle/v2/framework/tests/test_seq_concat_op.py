import unittest
import numpy as np
import sys
from op_test import OpTest


class TestConcatOp(OpTest):
    def set_data(self):
        # two level, batch size is 3
        x0 = np.random.random((4, 6, 3)).astype('float32')
        lod0 = [[0, 2, 4], [0, 1, 2, 3, 4]]
        x1 = np.random.random((4, 8, 3)).astype('float32')
        lod1 = [[0, 2, 4], [0, 1, 2, 3, 4]]
        axis = 1
        level = 1
        self.inputs = {'X': [('x0', (x0, lod0)), ('x1', (x1, lod1))]}
        self.attrs = {'axis': axis, 'level': level}
        outs = []
        for i in range(4):
            sub_x0 = x0[lod0[level][i]:lod0[level][i + 1], :]
            sub_x1 = x1[lod1[level][i]:lod1[level][i + 1], :]
            outs.append(np.concatenate((sub_x0, sub_x1), axis=axis))

        self.outputs = {'Out': np.concatenate(outs, axis=0)}

    def setUp(self):
        self.op_type = "sequence_concat"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')


class TestConcatOpDiffLod(TestConcatOp):
    def set_data(self):
        # two level, batch size is 3
        x0 = np.random.random((4, 6, 3)).astype('float32')
        lod0 = [[0, 2, 4], [0, 1, 2, 3, 4]]
        x1 = np.random.random((5, 6, 3)).astype('float32')
        lod1 = [[0, 3, 5], [0, 1, 2, 3, 5]]
        axis = 0
        level = 1
        self.inputs = {'X': [('x0', (x0, lod0)), ('x1', (x1, lod1))]}
        self.attrs = {'axis': axis, 'level': level}
        outs = []
        for i in range(4):
            sub_x0 = x0[lod0[level][i]:lod0[level][i + 1], :]
            sub_x1 = x1[lod1[level][i]:lod1[level][i + 1], :]
            outs.append(np.concatenate((sub_x0, sub_x1), axis=axis))

        self.outputs = {'Out': np.concatenate(outs, axis=0)}


class TestConcatOpLevelZero(TestConcatOp):
    def set_data(self):
        # two level, batch size is 3
        x0 = np.random.random((4, 3, 4)).astype('float32')
        lod0 = [[0, 2, 4], [0, 1, 2, 3, 4]]
        x1 = np.random.random((5, 3, 4)).astype('float32')
        lod1 = [[0, 3, 5], [0, 1, 3, 4, 5]]
        axis = 0
        level = 0
        self.inputs = {'X': [('x0', (x0, lod0)), ('x1', (x1, lod1))]}
        self.attrs = {'axis': axis, 'level': level}
        outs = []
        for i in range(2):
            sub_x0 = x0[lod0[level][i]:lod0[level][i + 1], :]
            sub_x1 = x1[lod1[level][i]:lod1[level][i + 1], :]
            outs.append(np.concatenate((sub_x0, sub_x1), axis=axis))

        self.outputs = {'Out': np.concatenate(outs, axis=0)}


if __name__ == '__main__':
    sys.exit(0)
    unittest.main()
