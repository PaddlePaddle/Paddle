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
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle


class XPUTestEyeOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'eye'
        self.use_dynamic_create_class = False

    class TestXPUEyeOp(XPUOpTest):
        def setUp(self):
            self.op_type = "eye"
            self.dtype = self.in_type
            self.inputs = {}
            self.n = 100
            self.m = 100
            self.init_shape()
            self.attrs = {
                'num_rows': self.n,
                'num_columns': self.m,
            }
            if self.dtype == np.uint16:
                result = np.eye(self.n, self.m, dtype=np.float32)
                self.outputs = {'Out': convert_float_to_uint16(result)}
            else:
                result = np.eye(self.n, self.m, dtype=self.dtype)
                self.outputs = {'Out': result}

        def init_shape(self):
            self.n = 100
            self.m = 100

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    class TestXPUEyeOp1(TestXPUEyeOp):
        def init_shape(self):
            self.n = 1000
            self.m = 100

    class TestXPUEyeOp2(TestXPUEyeOp):
        def init_shape(self):
            self.n = 100
            self.m = 1000

    class TestXPUEyeOp3(TestXPUEyeOp):
        def init_shape(self):
            self.n = 99
            self.m = 101

    class TestXPUEyeOp4(TestXPUEyeOp):
        def init_shape(self):
            self.n = 2
            self.m = 2

    class TestXPUEyeOp5(TestXPUEyeOp):
        def init_shape(self):
            self.n = 67
            self.m = 67


support_types = get_xpu_op_support_types('eye')
for stype in support_types:
    create_test_class(globals(), XPUTestEyeOp, stype)

if __name__ == "__main__":
    unittest.main()
