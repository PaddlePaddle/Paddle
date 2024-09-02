#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import Program, program_guard

paddle.enable_static()


class XPUTestMeanOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'mean'
        self.use_dynamic_create_class = False

    class TestMeanOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "mean"
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_shape()
            self.init_data()
            if self.dtype == np.uint16:
                x_float32 = convert_uint16_to_float(self.inputs["X"])
                self.outputs = {"Out": np.mean(x_float32)}
            else:
                self.outputs = {"Out": np.mean(self.inputs["X"])}

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.dtype

        def init_shape(self):
            self.shape = (10, 10)

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

        def test_checkout_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestMeanOp1(TestMeanOp):
        def set_shape(self):
            self.shape = 5

    class TestMeanOp2(TestMeanOp):
        def set_shape(self):
            self.shape = (5, 7, 8)

    class TestMeanOp3(TestMeanOp):
        def set_shape(self):
            self.shape = (10, 5, 7, 8)

    class TestMeanOp4(TestMeanOp):
        def set_shape(self):
            self.shape = (2, 2, 3, 3, 3)


class TestMeanOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of mean_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, paddle.mean, input1)
            # The input dtype of mean_op must be float16, float32, float64.
            input3 = paddle.static.data(
                name='input3', shape=[-1, 4], dtype="float16"
            )
            paddle.nn.functional.softmax(input3)


support_types = get_xpu_op_support_types('mean')
for stype in support_types:
    create_test_class(globals(), XPUTestMeanOp, stype)

if __name__ == "__main__":
    unittest.main()
