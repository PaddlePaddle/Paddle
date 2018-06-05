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
from op_test import OpTest


class TestSequenceReshape(OpTest):
    def setUp(self):
        self.op_type = 'sequence_reshape'
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = np.random.uniform(0.1, 1, [11, 24]).astype('float32')

        self.inputs = {'X': (x, x_lod)}
        self.attrs = {'new_dim': dimension}

        out, out_lod = self.compute_output(x, x_lod, dimension)

        self.outputs = {'Out': (out, out_lod)}

    def compute_output(self, x, x_lod, dimension):
        x_width = x.shape[1]
        out_lod = [[]]
        for i in xrange(len(x_lod[0])):
            seq_len = x_lod[0][i]
            offset = (seq_len * x_width) / dimension
            assert int(offset) * dimension == seq_len * x_width
            out_lod[0].append(int(offset))
        out = np.zeros(shape=(sum(out_lod[0]), dimension)).astype('float32')
        out.ravel()[:] = x.ravel()[:]
        return out, out_lod

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestSequenceReshape_reduce(TestSequenceReshape):
    def setUp(self):
        self.op_type = 'sequence_reshape'
        dimension = 24
        x_lod = [[4, 2, 2, 4]]
        x = np.random.uniform(0.1, 1, [12, 12]).astype('float32')

        self.inputs = {'X': (x, x_lod)}
        self.attrs = {'new_dim': dimension}

        out, out_lod = self.compute_output(x, x_lod, dimension)

        self.outputs = {'Out': (out, out_lod)}


class TestSequenceReshape_same(TestSequenceReshape):
    def setUp(self):
        self.op_type = 'sequence_reshape'
        dimension = 12
        x_lod = [[4, 2, 2, 4]]
        x = np.random.uniform(0.1, 1, [12, 12]).astype('float32')

        self.inputs = {'X': (x, x_lod)}
        self.attrs = {'new_dim': dimension}

        out, out_lod = self.compute_output(x, x_lod, dimension)

        self.outputs = {'Out': (out, out_lod)}


if __name__ == '__main__':
    unittest.main()
