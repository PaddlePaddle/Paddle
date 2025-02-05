#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def common_setup(self, index_num, nshards, shard_id, ignore_value):
    self.op_type = 'shard_index'
    self.python_api = paddle.tensor.shard_index
    x_lod = [list(range(10))]
    N = sum(x_lod[0])
    x = [np.random.randint(0, index_num - 1) for i in range(N)]
    x = np.array(x).astype('int32').reshape([N, 1])

    shard_size = (index_num + nshards - 1) // nshards
    out = np.zeros(shape=x.shape).astype('int32')
    for i in range(N):
        if x[i] // shard_size == shard_id:
            out[i] = x[i] % shard_size
        else:
            out[i] = ignore_value

    self.inputs = {'X': (x, x_lod)}
    self.attrs = {
        'index_num': index_num,
        'nshards': nshards,
        'shard_id': shard_id,
        'ignore_value': ignore_value,
    }
    self.outputs = {'Out': (out, x_lod)}


class TestShardIndexShardId0Op(OpTest):
    def setUp(self):
        common_setup(self, 20, 2, 0, -1)

    def test_check_output(self):
        self.check_output()


class TestShardIndexShardId1Op(OpTest):
    def setUp(self):
        common_setup(self, 20, 2, 1, -1)

    def test_check_output(self):
        self.check_output()


class TestShardIndexIgnoreValueOp(OpTest):
    def setUp(self):
        common_setup(self, 20, 2, 0, -2)

    def test_check_output(self):
        self.check_output()


class TestShardIndexNotEvenlyDividedOp(OpTest):
    def setUp(self):
        common_setup(self, 15, 2, 1, -1)

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
