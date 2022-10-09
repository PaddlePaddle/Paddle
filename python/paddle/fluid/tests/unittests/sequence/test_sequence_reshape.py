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
import math
import sys

sys.path.append("../")
from op_test import OpTest

import paddle.fluid as fluid


class TestSequenceReshape(OpTest):

    def init_data(self):
        self.dimension = 12
        self.x_lod = [[4, 1, 3, 3]]
        self.x = np.random.uniform(0.1, 1, [11, 24]).astype('float64')

    def setUp(self):
        self.init_data()
        self.op_type = 'sequence_reshape'
        self.inputs = {'X': (self.x, self.x_lod)}
        self.attrs = {'new_dim': self.dimension}
        out, out_lod = self.compute_output(self.x, self.x_lod, self.dimension)
        self.outputs = {'Out': (out, out_lod)}

    def compute_output(self, x, x_lod, dimension):
        x_width = x.shape[1]
        out_lod = [[]]
        for i in range(len(x_lod[0])):
            seq_len = x_lod[0][i]
            offset = (seq_len * x_width) / dimension
            assert int(offset) * dimension == seq_len * x_width
            out_lod[0].append(int(offset))
        out = np.zeros(shape=(sum(out_lod[0]), dimension)).astype('float64')
        out.ravel()[:] = x.ravel()[:]
        return out, out_lod

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestSequenceReshape_reduce(TestSequenceReshape):

    def init_data(self):
        self.dimension = 24
        self.x_lod = [[4, 2, 2, 4]]
        self.x = np.random.uniform(0.1, 1, [12, 12]).astype('float64')


class TestSequenceReshape_same(TestSequenceReshape):

    def init_data(self):
        self.dimension = 12
        self.x_lod = [[4, 2, 2, 4]]
        self.x = np.random.uniform(0.1, 1, [12, 12]).astype('float64')


class TestSequenceReshape_reduce_seq_len0(TestSequenceReshape):

    def init_data(self):
        self.dimension = 24
        self.x_lod = [[0, 6, 0, 2, 4]]
        self.x = np.random.uniform(0.1, 1, [12, 12]).astype('float64')


class TestSequenceReshape_reduce_seq_len0_case1(TestSequenceReshape):

    def init_data(self):
        self.dimension = 24
        self.x_lod = [[0, 2, 8, 2, 0]]
        self.x = np.random.uniform(0.1, 1, [12, 12]).astype('float64')


class TestSequenceReshapeOpError(unittest.TestCase):

    def test_error(self):

        def test_variable():
            x = np.random.random((2, 4)).astype("float32")
            fluid.layers.sequence_reshape(x=x, new_dim=4)

        self.assertRaises(TypeError, test_variable)

        def test_dtype():
            x1 = fluid.layers.data(name='x1',
                                   shape=[2, 6],
                                   append_batch_size=False,
                                   dtype='float16',
                                   lod_level=1)
            fluid.layers.sequence_reshape(x=x1, new_dim=4)

        self.assertRaises(TypeError, test_dtype)


if __name__ == '__main__':
    unittest.main()
