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

import unittest
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest
import numpy as np


class TestSequenceReverseBase(OpTest):
    def initParameters(self):
        pass

    def setUp(self):
        self.size = (10, 3, 4)
        self.lod = [2, 3, 5]
        self.dtype = 'float32'
        self.initParameters()
        self.op_type = 'sequence_reverse'
        self.x = np.random.random(self.size).astype(self.dtype)
        self.y = self.get_output()

        self.inputs = {'X': (self.x, [self.lod, ]), }
        self.outputs = {'Y': (self.y, [self.lod, ]), }

    def get_output(self):
        tmp_x = np.reshape(self.x, newshape=[self.x.shape[0], -1])
        tmp_y = np.ndarray(tmp_x.shape).astype(self.dtype)
        prev_idx = 0
        for cur_len in self.lod:
            idx_range = range(prev_idx, prev_idx + cur_len)
            tmp_y[idx_range, :] = np.flip(tmp_x[idx_range, :], 0)
            prev_idx += cur_len

        return np.reshape(tmp_y, newshape=self.x.shape).astype(self.dtype)

    def test_output(self):
        self.check_output(0)

    def test_grad(self):
        self.check_grad(['X'], 'Y')


class TestSequenceReserve1(TestSequenceReverseBase):
    def initParameters(self):
        self.size = (12, 10)
        self.lod = [4, 5, 3]


class TestSequenceReverse2(TestSequenceReverseBase):
    def initParameters(self):
        self.size = (12, 10)
        self.lod = [12]


class TestSequenceReverse3(TestSequenceReverseBase):
    def initParameters(self):
        self.size = (12, 10)
        self.lod = [3, 0, 6, 3]


if __name__ == '__main__':
    unittest.main()
