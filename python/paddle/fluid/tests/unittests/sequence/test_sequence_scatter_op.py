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
import sys
sys.path.append("../")
from op_test import OpTest


class TestSequenceScatterOp(OpTest):
    def init_lod(self):
        return [[30, 50, 40]]

    def setUp(self):
        self.op_type = "sequence_scatter"

        X_data = np.random.uniform(0.1, 1.0, [3, 6]).astype('float64')
        Ids_data = np.random.randint(0, 6, (120, 1)).astype('int64')
        Ids_lod = self.init_lod()

        Updates_data = np.random.uniform(0.1, 1.0, [120, 1]).astype('float64')
        Updates_lod = Ids_lod

        Out_data = np.copy(X_data)
        offset = 0
        for i in range(3):
            for j in range(Ids_lod[0][i]):
                Out_data[i][Ids_data[offset + j]] += Updates_data[offset + j]
            offset += Ids_lod[0][i]

        self.inputs = {
            'X': X_data,
            'Ids': (Ids_data, Ids_lod),
            'Updates': (Updates_data, Updates_lod)
        }
        self.outputs = {'Out': Out_data}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(['Updates'], 'Out', in_place=True, check_dygraph=False)


class TestSequenceScatterOpSeqLen0(TestSequenceScatterOp):
    def init_lod(self):
        return [[60, 60, 00]]


class TestSequenceScatterOpSeqLen0Case1(TestSequenceScatterOp):
    def init_lod(self):
        return [[0, 60, 60]]


class TestSequenceScatterOpSeqLen0Case2(TestSequenceScatterOp):
    def init_lod(self):
        return [[60, 0, 60]]


class TestSequenceScatterOpSeqLen0Case3(TestSequenceScatterOp):
    def init_lod(self):
        return [[120, 0, 0]]


class TestSequenceScatterOpSeqLen0Case4(TestSequenceScatterOp):
    def init_lod(self):
        return [[0, 120, 0]]


class TestSequenceScatterOpSeqLen0Case5(TestSequenceScatterOp):
    def init_lod(self):
        return [[0, 0, 120]]


# run the uni tests
if __name__ == "__main__":
    unittest.main()
