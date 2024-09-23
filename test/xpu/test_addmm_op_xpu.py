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
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestAddMMOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "addmm"
        self.use_dynamic_create_class = False

    class TestAddMMOp(XPUOpTest):
        """
        case 1
        """

        def setUp(self):
            self.op_type = "addmm"
            self.dtype = self.in_type
            self.init_case()
            if self.dtype == np.uint16:
                self.input_fp32 = np.random.random(self.input_shape).astype(
                    np.float32
                )
                self.x_fp32 = np.random.random(self.x_shape).astype(np.float32)
                self.y_fp32 = np.random.random(self.y_shape).astype(np.float32)
                self.input = convert_float_to_uint16(self.input_fp32)
                self.x = convert_float_to_uint16(self.x_fp32)
                self.y = convert_float_to_uint16(self.y_fp32)
                dot_result = np.dot(self.x_fp32, self.y_fp32)
                self.outputs = {
                    'Out': convert_float_to_uint16(
                        self.beta * self.input_fp32 + self.alpha * dot_result
                    )
                }
            else:
                self.input = np.random.random(self.input_shape).astype(
                    self.dtype
                )
                self.x = np.random.random(self.x_shape).astype(self.dtype)
                self.y = np.random.random(self.y_shape).astype(self.dtype)
                dot_result = np.dot(self.x, self.y)
                self.outputs = {
                    'Out': self.beta * self.input + self.alpha * dot_result
                }
            self.inputs = {
                'Input': self.input,
                'X': self.x,
                'Y': self.y,
            }
            self.attrs = {
                'Alpha': self.alpha,
                'Beta': self.beta,
            }

        def init_case(self):
            self.input_shape = [10, 10]
            self.x_shape = [10, 10]
            self.y_shape = [10, 10]
            # self.alpha = 1.0
            # self.beta = 1.0
            self.alpha = 0.0
            self.beta = 1.0

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

        def test_check_grad(self):
            if (
                hasattr(self.__class__, "no_need_check_grad")
                and self.__class__.no_need_check_grad
            ):
                return
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['Input', 'X', 'Y'], 'Out')

    '''
    class TestAddMMOp2(TestAddMMOp):
        """
        case 2
        """

        def init_case(self):
            self.input_shape = [11, 11]
            self.x_shape = [11, 13]
            self.y_shape = [13, 11]
            self.alpha = 1.0
            self.beta = 1.0

    class TestAddMMOp3(TestAddMMOp):
        """
        case 3
        """

        def init_case(self):
            self.input_shape = [11, 11]
            self.x_shape = [11, 13]
            self.y_shape = [13, 11]
            self.alpha = 1.0
            self.beta = 1.0

    class TestAddMMOp4(TestAddMMOp):
        """
        case 4
        """

        def init_case(self):
            self.input_shape = [11, 13]
            self.x_shape = [11, 15]
            self.y_shape = [15, 13]
            self.alpha = 1.0
            self.beta = 1.0

    class TestAddMMOp5(TestAddMMOp):
        """
        case 5
        """

        def init_case(self):
            self.input_shape = [11, 13]
            self.x_shape = [11, 15]
            self.y_shape = [15, 13]
            self.alpha = 0.0
            self.beta = 1.0

    class TestAddMMOp6(TestAddMMOp):
        """
        case 6
        """

        def init_case(self):
            self.input_shape = [11, 13]
            self.x_shape = [11, 15]
            self.y_shape = [15, 13]
            self.alpha = 1.0
            self.beta = 0.0

    class TestAddMMOp7(TestAddMMOp):
        """
        case 7
        """

        def init_case(self):
            self.input_shape = [11, 13]
            self.x_shape = [11, 15]
            self.y_shape = [15, 13]
            self.alpha = 0.0
            self.beta = 0.0
    '''


# support_types = get_xpu_op_support_types('addmm')
support_types = ['float32']
for stype in support_types:
    create_test_class(globals(), XPUTestAddMMOp, stype)

if __name__ == "__main__":
    unittest.main()
