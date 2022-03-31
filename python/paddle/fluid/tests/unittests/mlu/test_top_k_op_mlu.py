#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append('..')
from op_test import OpTest
import paddle
import paddle.fluid.core as core

paddle.enable_static()


class TestTopkOp(OpTest):
    def setUp(self):
        self.variable_k = False
        self.set_args()
        self.op_type = "top_k"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.__class__.no_need_check_grad = True
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
        self.row = 100
        self.top_k = 1

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestTopkFP16Op(TestTopkOp):
    def init_dtype(self):
        self.dtype = np.float16


if __name__ == "__main__":
    unittest.main()
