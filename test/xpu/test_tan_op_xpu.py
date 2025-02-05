# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import convert_float_to_uint16, convert_uint16_to_float
from op_test_xpu import XPUOpTest

import paddle


class XPUTestTanOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "tan"
        self.use_dynamic_create_class = False

    class TestTanOp(XPUOpTest):
        def setUp(self):
            self.op_type = "tan"
            self.init_shape()
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.init_data()

        def init_shape(self):
            self.shape = [4, 10, 10]

        def init_data(self):
            x = np.random.uniform(-1, 1, self.shape).astype("float32")
            if self.dtype == np.uint16:
                x = convert_float_to_uint16(x)
            else:
                x = x.astype(self.dtype)

            self.inputs = {'X': x}

            if self.dtype == np.uint16:
                out = convert_float_to_uint16(
                    np.tan(convert_uint16_to_float(x))
                )
            else:
                out = np.tan(x)
            self.outputs = {'Out': out}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestTanOp1(TestTanOp):
        def init_shape(self):
            self.shape = [8, 16, 8]

    class TestTanOp2(TestTanOp):
        def init_shape(self):
            self.shape = [8, 16]

    class TestTanOp3(TestTanOp):
        def init_shape(self):
            self.shape = [4, 8, 17]

    class TestTanOp4(TestTanOp):
        def init_shape(self):
            self.shape = [4, 9, 8]

    class TestTanOp5(TestTanOp):
        def init_shape(self):
            self.shape = [17]


support_types = get_xpu_op_support_types('tan')
for stype in support_types:
    create_test_class(globals(), XPUTestTanOp, stype)


if __name__ == "__main__":
    unittest.main()
