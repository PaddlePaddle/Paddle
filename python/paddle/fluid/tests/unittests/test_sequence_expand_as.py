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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest


class TestSequenceExpandAs(OpTest):
    def setUp(self):
        self.op_type = 'sequence_expand_as'
        self.set_data()
        self.compute()

    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [3, 1]).astype('float32')
        y_data = np.random.uniform(0.1, 1, [8, 1]).astype('float32')
        y_lod = [[1, 3, 4]]
        self.inputs = {'X': x_data, 'Y': (y_data, y_lod)}

    def compute(self):
        x = self.inputs['X']
        x_data, x_lod = x if type(x) == tuple else (x, None)
        y_data, y_lod = self.inputs['Y']

        assert len(y_lod) == 1 and len(y_lod[0]) == x_data.shape[0]

        repeats = []
        for i in range(len(y_lod[0])):
            repeat_num = y_lod[0][i]
            if repeat_num == 0:
                continue
            repeats.extend([i for _ in range(repeat_num)])

        out_data = x_data[repeats]
        self.outputs = {'Out': (out_data, y_lod)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestSequenceExpandAsCase1(TestSequenceExpandAs):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [5, 1]).astype('float32')
        x_lod = [[2, 3]]
        y_data = np.random.uniform(0.1, 1, [10, 1]).astype('float32')
        y_lod = [[2, 2, 0, 3, 3]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}


class TestSequenceExpandAsCase2(TestSequenceExpandAs):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [5, 1]).astype('float32')
        x_lod = [[2, 3]]
        y_data = np.random.uniform(0.1, 1, [10, 1]).astype('float32')
        y_lod = [[0, 4, 0, 6, 0]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}


class TestSequenceExpandAsCase3(TestSequenceExpandAs):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [1, 2, 2]).astype('float32')
        x_lod = [[1]]
        y_data = np.random.uniform(0.1, 1, [2, 2, 2]).astype('float32')
        y_lod = [[2]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}


if __name__ == '__main__':
    unittest.main()
