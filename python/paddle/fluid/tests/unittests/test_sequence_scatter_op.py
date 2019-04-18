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


class TestSequenceScatterOp(OpTest):
    def init_lod(self):
        return [[3, 5, 4]]

    def setUp(self):
        self.op_type = "sequence_scatter"

        X_data = np.random.uniform(0.1, 1.0, [3, 6]).astype('float32')
        Ids_data = np.array([[0], [1], [2], [5], [4], [3], [0], [1], [3], [2],
                             [5], [4]]).astype('int64')
        Ids_lod = self.init_lod()

        Updates_data = np.random.uniform(0.1, 1.0, [12, 1]).astype('float32')
        Updates_lod = Ids_lod

        Out_data = np.copy(X_data)
        offset = 0
        for i in xrange(3):
            Out_data[i][Ids_data[offset:(offset + Ids_lod[0][
                i])]] += Updates_data[offset:(offset + Ids_lod[0][i])]
            offset += Ids_lod[0][i]

        self.inputs = {
            'X': X_data,
            'Ids': (Ids_data, Ids_lod),
            'Updates': (Updates_data, Updates_lod)
        }
        self.outputs = {'Out': Out_data}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Updates'], 'Out', in_place=True)


class TestSequenceScatterOpSeqLen0(TestSequenceScatterOp):
    def init_lod(self):
        return [[6, 0, 6]]


class TestSequenceScatterOpSeqLen0Case1(TestSequenceScatterOp):
    def init_lod(self):
        return [[0, 6, 6]]


class TestSequenceScatterOpSeqLen0Case2(TestSequenceScatterOp):
    def init_lod(self):
        return [[6, 6, 0]]


if __name__ == "__main__":
    unittest.main()
