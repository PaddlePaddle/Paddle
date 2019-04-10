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


class TestTopkOp(OpTest):
    def setUp(self):
        self.variable_k = False
        self.set_args()
        self.op_type = "top_k"
        self.dtype = np.float32
        self.init_dtype()

        k = self.top_k
        input = np.random.random((self.row, k)).astype(self.dtype)
        output = np.ndarray((self.row, k))
        indices = np.ndarray((self.row, k)).astype("int64")
        self.inputs = {'X': input}

        if self.variable_k:
            self.inputs['K'] = np.array([k]).astype("int32")
        else:
            self.attrs = {'k': k}

        for rowid in range(self.row):
            row = input[rowid]
            output[rowid] = np.sort(row)[::-1][:k]
            indices[rowid] = row.argsort()[::-1][:k]

        self.outputs = {'Out': output, 'Indices': indices}

    def init_dtype(self):
        pass

    def set_args(self):
        self.row = 32
        self.top_k = 1

    def test_check_output(self):
        self.check_output()


class TestTopkOpFp16(TestTopkOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestTopkOp3d(OpTest):
    def setUp(self):
        self.op_type = "top_k"
        k = 1
        input = np.random.random((32, 2, 84)).astype("float32")
        input_flat_2d = input.reshape(64, 84)
        output = np.ndarray((64, k))
        indices = np.ndarray((64, k)).astype("int64")

        self.inputs = {'X': input}
        self.attrs = {'k': k}

        for rowid in range(64):
            row = input_flat_2d[rowid]
            output[rowid] = np.sort(row)[::-1][:k]
            indices[rowid] = row.argsort()[::-1][:k]

        self.outputs = {
            'Out': output.reshape((32, 2, k)),
            'Indices': indices.reshape((32, 2, k))
        }

    def test_check_output(self):
        self.check_output()


class TestTopkOp2(OpTest):
    def setUp(self):
        self.op_type = "top_k"
        k = 1
        m = 2056
        input = np.random.random((m, 84)).astype("float32")
        output = np.ndarray((m, k))
        indices = np.ndarray((m, k)).astype("int64")

        self.inputs = {'X': input}
        self.attrs = {'k': k}

        for rowid in range(m):
            row = input[rowid]
            output[rowid] = -np.sort(-row)[:k]
            indices[rowid] = (-row).argsort()[:k]

        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        self.check_output()


class TestTopkOp3(TestTopkOp):
    def set_args(self):
        self.row = 2056
        self.top_k = 3


class TestTopkOp4(TestTopkOp):
    def set_args(self):
        self.row = 40000
        self.top_k = 1


class TestTopkOp5(TestTopkOp):
    def set_args(self):
        self.row = 40000
        self.top_k = 3
        self.variable_k = True


if __name__ == "__main__":
    unittest.main()
