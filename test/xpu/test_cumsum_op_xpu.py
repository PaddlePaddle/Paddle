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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestCumsumOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'cumsum'
        self.use_dynamic_create_class = False

    class TestCumsumOPBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.set_case()

        def set_case(self):
            self.op_type = 'cumsum'
            self.init_config()

            if self.dtype == np.uint16:
                self.data = np.random.uniform(
                    -1.0, 1.0, self.input_shape
                ).astype('float32')
                self.inputs = {'X': convert_float_to_uint16(self.data)}
            else:
                self.data = np.random.uniform(
                    -1.0, 1.0, self.input_shape
                ).astype(self.dtype)
                self.inputs = {
                    'X': self.data,
                }
            reference_out = np.cumsum(self.data, axis=self.axis)
            self.attrs = {
                'use_xpu': True,
                'axis': self.axis,
                'flatten': True if self.axis is None else False,
            }
            self.outputs = {'Out': reference_out}

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_config(self):
            self.input_shape = (2, 5)
            self.axis = None

    class XPUTestCumsum1(TestCumsumOPBase):
        def init_config(self):
            self.input_shape = [2, 768]
            self.axis = 0

    class XPUTestCumsum2(TestCumsumOPBase):
        def init_config(self):
            self.input_shape = [3, 8, 4096]
            self.axis = 1

    class XPUTestCumsum3(TestCumsumOPBase):
        def init_config(self):
            self.input_shape = [1024]
            self.axis = 0

    class XPUTestCumsum4(TestCumsumOPBase):
        def init_config(self):
            self.input_shape = [2, 2, 255]
            self.axis = -1

    class XPUTestCumsum5(TestCumsumOPBase):
        def init_config(self):
            self.input_shape = [2, 3, 5, 255]
            self.axis = 2


support_types = get_xpu_op_support_types('cumsum')
for stype in support_types:
    create_test_class(globals(), XPUTestCumsumOP, stype)

if __name__ == "__main__":
    unittest.main()
