#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def api_wrapper(x, k):
    return paddle._legacy_C_ops.top_k(x, "k", k)


class XPUTestTopKOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'top_k'
        self.use_dynamic_create_class = False

    class TestTopkOp(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.variable_k = False
            self.op_type = "top_k"
            self.python_api = api_wrapper
            self.init_args()
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
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_args(self):
            self.row = 100
            self.top_k = 1


support_types = get_xpu_op_support_types('top_k')
for stype in support_types:
    create_test_class(globals(), XPUTestTopKOp, stype)

if __name__ == "__main__":
    unittest.main()
