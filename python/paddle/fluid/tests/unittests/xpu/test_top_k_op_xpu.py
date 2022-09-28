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

import unittest
import numpy as np
import sys

sys.path.append("..")
import paddle
from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


def random_unique_float(row, k, dtype):
    # create a random float array with 10x length
    arr = np.random.uniform(-10.0, 10.0, int(row * k * 10)).astype(dtype)
    arr = np.unique(arr)
    assert arr.shape[
        0] >= row * k, "failed to create enough unique values: %d vs %d" % (
            arr.shape[0], row * k)
    arr = arr[:row * k]
    np.random.shuffle(arr)
    arr = arr.reshape(row, k)
    return arr


class XPUTestTopkOP(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'top_k'
        self.use_dynamic_create_class = False

    class TestXPUTopkOP(XPUOpTest):

        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.op_type = 'top_k'
            self.set_case()

            # generate UNIQUE float values as input, in order to prevent the following potential problem: x[i] and x[j] are IDENTICAL float values, the result of cpu index is [i, j] while the xpu result is [j, i]. Both of them are correct but diff in numpy compare.
            k = self.top_k
            input = random_unique_float(self.row, k, self.dtype)
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

        def set_case(self):
            self.variable_k = False
            self.row = 16
            self.top_k = 8

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestTopk1(TestXPUTopkOP):

        def set_case(self):
            self.variable_k = True
            self.row = 100
            self.top_k = 1

    class TestTopk2(TestXPUTopkOP):

        def set_case(self):
            self.variable_k = False
            self.row = 16
            self.top_k = 256

    class TestTopk3(TestXPUTopkOP):

        def set_case(self):
            self.variable_k = True
            self.row = 10
            self.top_k = 512

    class TestTopk4(TestXPUTopkOP):

        def set_case(self):
            self.variable_k = False
            self.row = 5
            self.top_k = 511


support_types = get_xpu_op_support_types('top_k')
for stype in support_types:
    create_test_class(globals(), XPUTestTopkOP, stype)

if __name__ == "__main__":
    unittest.main()
