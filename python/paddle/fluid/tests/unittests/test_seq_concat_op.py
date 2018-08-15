#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import sys
from op_test import OpTest


def to_abs_offset_lod(lod):
    offset_lod = [[0] for i in lod]
    for i, level in enumerate(lod):
        for seq_len in level:
            offset_lod[i].append(offset_lod[i][-1] + seq_len)

    if len(offset_lod) == 0 or len(offset_lod) == 1:
        return offset_lod
    import copy
    new_offset_lod = copy.deepcopy(offset_lod)
    for idx, val in enumerate(offset_lod[0]):
        new_offset_lod[0][idx] = offset_lod[1][val]
    return new_offset_lod


def seq_concat(inputs, level):
    lod0 = inputs['X'][0][1][1]
    lod1 = inputs['X'][1][1][1]
    x0 = inputs['X'][0][1][0]
    x1 = inputs['X'][1][1][0]
    level_idx = len(lod0) - level - 1
    outs = []
    for i in range(len(lod0[level_idx])):
        sub_x0 = x0[to_abs_offset_lod(lod0)[level_idx][i]:to_abs_offset_lod(
            lod0)[level_idx][i + 1], :]
        sub_x1 = x1[to_abs_offset_lod(lod1)[level_idx][i]:to_abs_offset_lod(
            lod1)[level_idx][i + 1], :]
        outs.append(np.concatenate((sub_x0, sub_x1), axis=0))
    return np.concatenate(outs, axis=0)


class TestSeqConcatOp(OpTest):
    def set_data(self):
        # two level, batch size is 3
        x0 = np.random.random((4, 6, 3)).astype('float32')
        lod0 = [[2, 2], [1, 1, 1, 1]]
        x1 = np.random.random((4, 8, 3)).astype('float32')
        lod1 = [[2, 2], [1, 1, 1, 1]]
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
        lod0 = [[2, 2], [1, 1, 1, 1]]
        x1 = np.random.random((7, 6, 3)).astype('float32')
        lod1 = [[2, 2], [1, 2, 2, 2]]
        axis = 0
        level = 0
        self.inputs = {'X': [('x0', (x0, lod0)), ('x1', (x1, lod1))]}
        self.attrs = {'axis': axis, 'level': level}
        out_lod = [[2, 2], [2, 3, 3, 3]]
        self.outputs = {'Out': (seq_concat(self.inputs, level), out_lod)}


class TestSeqConcatOplevelOneNestedSequence(TestSeqConcatOp):
    def set_data(self):
        # two level, batch size is 3
        x0 = np.random.random((4, 6, 3)).astype('float32')
        lod0 = [[2, 2], [1, 1, 1, 1]]
        x1 = np.random.random((7, 6, 3)).astype('float32')
        lod1 = [[3, 1], [1, 2, 2, 2]]
        axis = 0
        level = 1
        self.inputs = {'X': [('x0', (x0, lod0)), ('x1', (x1, lod1))]}
        self.attrs = {'axis': axis, 'level': level}
        out_lod = [[5, 3], [1, 1, 1, 2, 2, 1, 1, 2]]
        self.outputs = {'Out': (seq_concat(self.inputs, level), out_lod)}


class TestSeqConcatOpLevelZeroSequence(TestSeqConcatOp):
    def set_data(self):
        # two level, batch size is 3
        x0 = np.random.random((4, 3, 4)).astype('float32')
        lod0 = [[1, 1, 1, 1]]
        x1 = np.random.random((7, 3, 4)).astype('float32')
        lod1 = [[1, 2, 2, 2]]
        axis = 0
        level = 0
        self.inputs = {'X': [('x0', (x0, lod0)), ('x1', (x1, lod1))]}
        self.attrs = {'axis': axis, 'level': level}
        out_lod = [[2, 3, 3, 3]]
        self.outputs = {'Out': (seq_concat(self.inputs, level), out_lod)}


if __name__ == '__main__':
    unittest.main()
