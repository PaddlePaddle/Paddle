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
from op_test import OpTest


class TestSequenceExpand(OpTest):
    def set_data(self):
        x = [i / 10.0 for i in range(3)]
        y = [i / 10.0 for i in range(8)]
        x_data = np.array(x).reshape(3, 1).astype('float32')
        y_data = np.array(y).reshape(8, 1).astype('float32')
        print(x_data)
        print(y_data)
        # x_data = np.random.uniform(0.1, 1, [3, 1]).astype('float32')
        # y_data = np.random.uniform(0.1, 1, [8, 1]).astype('float32')
        y_lod = [[0, 1, 4, 8]]
        self.inputs = {'X': x_data, 'Y': (y_data, y_lod)}

    def compute(self):
        x = self.inputs['X']
        x_data, x_lod = x if type(x) == tuple else (x, None)
        y_data, y_lod = self.inputs['Y']

        if hasattr(self, 'attrs'):
            ref_level = self.attrs['ref_level']
        else:
            ref_level = len(y_lod) - 1

        out = np.zeros(shape=((0, ) + x_data.shape[1:]), dtype=x_data.dtype)

        if x_lod is None:
            x_idx = [i for i in xrange(x_data.shape[0] + 1)]
        else:
            x_idx = x_lod[0]
            out_lod = [[0]]

        for i in xrange(1, len(y_lod[ref_level])):
            repeat_num = y_lod[ref_level][i] - y_lod[ref_level][i - 1]
            x_len = x_idx[i] - x_idx[i - 1]
            if repeat_num > 0:
                x_sub = x_data[x_idx[i - 1]:x_idx[i], :]
                x_sub = np.repeat(x_sub, repeat_num, axis=0)
                out = np.vstack((out, x_sub))
                if x_lod is not None:
                    for j in xrange(repeat_num):
                        out_lod[0].append(out_lod[0][-1] + x_len)

        if x_lod is None:
            self.outputs = {'Out': out}
        else:
            self.outputs = {'Out': (out, out_lod)}

    def setUp(self):
        self.op_type = 'sequence_expand'
        self.set_data()
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestSequenceExpandCase1(TestSequenceExpand):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [5, 1]).astype('float32')
        x_lod = [[0, 2, 5]]
        y_data = np.random.uniform(0.1, 1, [13, 1]).astype('float32')
        y_lod = [[0, 2, 5], [0, 2, 4, 7, 10, 13]]
        self.inputs = {'X': x_data, 'Y': (y_data, y_lod)}
        self.attrs = {'ref_level': 0}


class TestSequenceExpandCase2(TestSequenceExpand):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [1, 2, 2]).astype('float32')
        x_lod = [[0, 1]]
        y_data = np.random.uniform(0.1, 1, [2, 2, 2]).astype('float32')
        y_lod = [[0, 2], [0, 2]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}
        self.attrs = {'ref_level': 0}


class TestSequenceExpandCase3(TestSequenceExpand):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [4, 1]).astype('float32')
        x_lod = [[0, 1, 2, 3, 4]]
        y_data = np.random.uniform(0.1, 1, [6, 1]).astype('float32')
        y_lod = [[0, 2, 4, 4, 6]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}


class TestSequenceExpandCase4(TestSequenceExpand):
    def set_data(self):
        data = [0.1, 0.3, 0.2, 0.15, 0.25, 0.2, 0.15, 0.25, 0.1, 0.3]
        x_data = np.array(data).reshape([5, 2]).astype('float32')
        x_lod = [[0, 2, 5]]
        y_data = np.random.uniform(0.1, 1, [2, 1]).astype('float32')
        y_lod = [[0, 1, 2], [0, 1, 2]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}


if __name__ == '__main__':
    unittest.main()
