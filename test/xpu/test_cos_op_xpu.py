#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


class XPUTestCosOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'cos'
        self.use_dynamic_create_class = False

    class TestCosOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "cos"
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_shape()
            self.init_data()
            if self.dtype == np.uint16:
                x_float32 = convert_uint16_to_float(self.inputs['X'])
                self.outputs = {'Out': np.cos(x_float32)}
            else:
                self.outputs = {'Out': np.cos(self.inputs['X'])}

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.dtype

        def init_shape(self):
            self.shape = (4, 10, 10)

        def init_data(self):
            if self.dtype == np.uint16:
                x = np.random.random(self.shape).astype('float32')
                x = convert_float_to_uint16(x)
                self.inputs = {'X': x}
            else:
                self.inputs = {
                    'X': np.random.random(self.shape).astype(self.dtype)
                }

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if core.is_compiled_with_xpu():
                self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestCosOp1(TestCosOp):
        def init_shape(self):
            self.shape = (8, 16, 8)

    class TestCosOp2(TestCosOp):
        def init_shape(self):
            self.shape = (8, 16)

    class TestCosOp3(TestCosOp):
        def init_shape(self):
            self.shape = (4, 8, 16)

    class TestCosOp4(TestCosOp):
        def init_shape(self):
            self.shape = (4, 8, 8)

    class TestCosOp5(TestCosOp):
        def init_shape(self):
            self.shape = (4, 8, 16)


support_types = get_xpu_op_support_types('cos')
for stype in support_types:
    create_test_class(globals(), XPUTestCosOp, stype)

if __name__ == '__main__':
    unittest.main()
