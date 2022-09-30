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
import paddle.fluid.core as core
import paddle


class TestTopkOp(OpTest):

    def setUp(self):
        self.variable_k = False
        self.set_args()
        self.op_type = "top_k"
        self.dtype = np.float64
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
        self.row = 100
        self.top_k = 1

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(set(['X']), 'Out')


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
