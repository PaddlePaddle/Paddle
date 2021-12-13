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
import numpy as np
import sys
sys.path.append("../")
from op_test import OpTest


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
        self.check_output(0, check_dygraph=False)

    def test_grad(self):
        self.check_grad(['X'], 'Y', check_dygraph=False)


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


class TestSequenceReverse4(TestSequenceReverseBase):
    def initParameters(self):
        self.size = (12, 10)
        self.lod = [0, 2, 10, 0]


class TestSequenceReverseOpError(unittest.TestCase):
    def test_error(self):
        def test_variable():
            # the input type must be Variable
            x_data = np.random.random((2, 4)).astype("float32")
            fluid.layers.sequence_reverse(x=x_data)

        self.assertRaises(TypeError, test_variable)

        def test_dtype():
            # dtype must be 'float32', 'float64', 'int8', 'int32', 'int64'
            x2_data = fluid.layers.data(name='x2', shape=[4], dtype='float16')
            fluid.layers.sequence_reverse(x=x2_data)

        self.assertRaises(TypeError, test_dtype)


if __name__ == '__main__':
    unittest.main()
