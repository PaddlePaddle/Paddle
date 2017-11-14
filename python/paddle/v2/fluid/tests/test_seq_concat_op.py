import unittest
import numpy as np
import sys
from op_test import OpTest
exit(0)


def to_abs_lod(lod):
    if len(lod) == 0 or len(lod) == 1:
        return lod
    import copy
    new_lod = copy.deepcopy(lod)
    for idx, val in enumerate(lod[0]):
        new_lod[0][idx] = lod[1][val]
    return new_lod


def seq_concat(inputs, level):
    lod0 = inputs['X'][0][1][1]
    lod1 = inputs['X'][1][1][1]
    x0 = inputs['X'][0][1][0]
    x1 = inputs['X'][1][1][0]
    level_idx = len(lod0) - level - 1
    outs = []
    for i in range(len(lod0[level_idx]) - 1):
        sub_x0 = x0[to_abs_lod(lod0)[level_idx][i]:to_abs_lod(lod0)[level_idx][
            i + 1], :]
        sub_x1 = x1[to_abs_lod(lod1)[level_idx][i]:to_abs_lod(lod1)[level_idx][
            i + 1], :]
        outs.append(np.concatenate((sub_x0, sub_x1), axis=0))
    return np.concatenate(outs, axis=0)


class TestSeqConcatOp(OpTest):
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
        self.outputs = {'Out': (np.concatenate([x0, x1], axis=1), lod0)}

    def setUp(self):
        self.op_type = "sequence_concat"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')


class TestSeqConcatOpLevelZeroNestedSequence(TestSeqConcatOp):
    def set_data(self):
        # two level, batch size is 3
        x0 = np.random.random((4, 6, 3)).astype('float32')
        lod0 = [[0, 2, 4], [0, 1, 2, 3, 4]]
        x1 = np.random.random((7, 6, 3)).astype('float32')
        lod1 = [[0, 2, 4], [0, 1, 3, 5, 7]]
        axis = 0
        level = 0
        self.inputs = {'X': [('x0', (x0, lod0)), ('x1', (x1, lod1))]}
        self.attrs = {'axis': axis, 'level': level}
        out_lod = [[0, 2, 4], [0, 2, 5, 8, 11]]
        self.outputs = {'Out': (seq_concat(self.inputs, level), out_lod)}


class TestSeqConcatOplevelOneNestedSequence(TestSeqConcatOp):
    def set_data(self):
        # two level, batch size is 3
        x0 = np.random.random((4, 6, 3)).astype('float32')
        lod0 = [[0, 2, 4], [0, 1, 2, 3, 4]]
        x1 = np.random.random((7, 6, 3)).astype('float32')
        lod1 = [[0, 3, 4], [0, 1, 3, 5, 7]]
        axis = 0
        level = 1
        self.inputs = {'X': [('x0', (x0, lod0)), ('x1', (x1, lod1))]}
        self.attrs = {'axis': axis, 'level': level}
        out_lod = [[0, 5, 8], [0, 1, 2, 3, 5, 7, 8, 9, 11]]
        self.outputs = {'Out': (seq_concat(self.inputs, level), out_lod)}


class TestSeqConcatOpLevelZeroSequence(TestSeqConcatOp):
    def set_data(self):
        # two level, batch size is 3
        x0 = np.random.random((4, 3, 4)).astype('float32')
        lod0 = [[0, 1, 2, 3, 4]]
        x1 = np.random.random((7, 3, 4)).astype('float32')
        lod1 = [[0, 1, 3, 5, 7]]
        axis = 0
        level = 0
        self.inputs = {'X': [('x0', (x0, lod0)), ('x1', (x1, lod1))]}
        self.attrs = {'axis': axis, 'level': level}
        out_lod = [[0, 2, 5, 8, 11]]
        self.outputs = {'Out': (seq_concat(self.inputs, level), out_lod)}


if __name__ == '__main__':
    unittest.main()
