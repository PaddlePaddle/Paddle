# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""This is unit test of Test shuffle_batch Op."""

from __future__ import print_function, division
import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from op_test import OpTest
import random


class TestShuffleBatchOp(OpTest):
    def setUp(self):
        self.op_type = 'shuffle_batch'
        self.dtype = np.float32
        x = np.random.random((5, 4)).astype(self.dtype)
        shuffle_order = np.arange(5)
        np.random.shuffle(shuffle_order)
        out = np.zeros((5, 4)).astype(self.dtype)
        for i in range(5):
            for j in range(4):
                out[shuffle_order[i]][j] = x[i][j]
        self.inputs = {'X': x, }
        self.outputs = {'Out': out, 'ShuffleIdx': shuffle_order}
        self.attrs = {'ShuffleOrder': shuffle_order}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestShuffleBatchOp2(OpTest):
    def setUp(self):
        self.op_type = 'shuffle_batch'
        self.dtype = np.float16
        x = np.random.random(5).astype(self.dtype)
        shuffle_order = np.arange(1)
        np.random.shuffle(shuffle_order)
        out = np.zeros(5).astype(self.dtype)
        for i in range(1):
            out[shuffle_order[i]] = x[i]
        self.inputs = {'X': x, }
        self.outputs = {'Out': out, 'ShuffleIdx': shuffle_order}
        self.attrs = {'ShuffleOrder': []}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestShuffleBatchOp3(OpTest):
    def setUp(self):
        self.op_type = 'shuffle_batch'
        self.dtype = np.float32
        x = np.random.random((5, 4, 2)).astype(self.dtype)
        shuffle_order = np.arange(5 * 4)
        np.random.shuffle(shuffle_order)
        out = np.zeros((5, 4, 2)).astype(self.dtype)
        for i in range(5):
            for j in range(4):
                idx = i * 4 + j
                new_idx = shuffle_order[idx]
                new_i, new_j = new_idx // 4, new_idx % 4
                for k in range(2):
                    out[new_i][new_j][k] = x[i][j][k]
        self.inputs = {'X': x, }
        self.outputs = {'Out': out, 'ShuffleIdx': shuffle_order}
        self.attrs = {'ShuffleOrder': shuffle_order}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestShuffleBatchOp4(OpTest):
    def setUp(self):
        self.op_type = 'shuffle_batch'
        self.dtype = np.double
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                      [11, 12]]).astype(self.dtype)
        x_rsl = [[2, 1, 3]]
        shuffle_order = np.array([3, 1, 5, 4, 2, 0])
        out = np.array([11, 12, 3, 4, 9, 10, 1, 2, 7, 8, 5, 6]).reshape(
            [6, 2]).astype(self.dtype)
        out_rsl = x_rsl
        self.inputs = {'X': (x, x_rsl), }
        self.outputs = {'Out': (out, out_rsl), 'ShuffleIdx': shuffle_order}
        self.attrs = {'ShuffleOrder': shuffle_order}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
