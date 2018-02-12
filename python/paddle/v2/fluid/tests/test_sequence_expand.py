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
        x_data = np.random.uniform(0.1, 1, [3, 1]).astype('float32')
        y_data = np.random.uniform(0.1, 1, [8, 1]).astype('float32')
        y_lod = [[0, 1, 4, 8]]
        self.inputs = {'X': x_data, 'Y': (y_data, y_lod)}

    def compute(self):
        x = self.inputs['X']
        x_data, x_lod = x if type(x) == tuple else (x, None)
        n = 1 + x_data.shape[0] if not x_lod else len(x_lod[0])
        y_data, y_lod = self.inputs['Y']
        repeats = [((y_lod[-1][i + 1] - y_lod[-1][i]))
                   for i in range(len(y_lod[-1]) - 1)]
        out = x_data.repeat(repeats, axis=0)
        self.outputs = {'Out': out}

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
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}


class TestSequenceExpandCase2(TestSequenceExpand):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [1, 2, 2]).astype('float32')
        x_lod = [[0, 1]]
        y_data = np.random.uniform(0.1, 1, [2, 2, 2]).astype('float32')
        y_lod = [[0, 2]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}


class TestSequenceExpandCase3(TestSequenceExpand):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [4, 1]).astype('float32')
        x_lod = [[0, 1, 2, 3, 4]]
        y_data = np.random.uniform(0.1, 1, [6, 1]).astype('float32')
        y_lod = [[0, 2, 4, 4, 6]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}


class TestSequenceExpandCase4(TestSequenceExpand):
    def set_data(self):
        x_data = np.array(
            [0.1, 0.3, 0.2, 0.15, 0.25, 0.2, 0.15, 0.25, 0.1, 0.3]).reshape(
                [2, 5]).astype('float32')
        x_lod = [[
            0,
            1,
            2,
        ]]
        y_data = np.random.uniform(0.1, 1, [2, 1]).astype('float32')
        y_lod = [[0, 1, 2], [0, 1, 2]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}


if __name__ == '__main__':
    unittest.main()
