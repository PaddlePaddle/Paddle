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

from __future__ import print_function

import sys

sys.path.append("..")
import unittest
import numpy as np

import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard, core
import paddle
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestConcatOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'concat'
        self.use_dynamic_create_class = False

    class TestConcatOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "concat"
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.init_axis()
            self.set_inputs()
            self.inputs = {
                'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]
            }
            self.attrs = {'axis': self.axis}
            if self.axis < 0:
                self.actual_axis = self.axis + len(self.x0.shape)
                self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
            else:
                self.actual_axis = self.axis

            self.outputs = {
                'Out': np.concatenate(
                    (self.x0, self.x1, self.x2), axis=self.actual_axis)
            }

        def set_inputs(self):
            self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
            self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
            self.x2 = np.random.random((2, 3, 4, 5)).astype(self.dtype)

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True

        def init_dtype(self):
            self.dtype = self.in_type

        def init_axis(self):
            self.axis = -1

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['x0'], 'Out')
                self.check_grad_with_place(place, ['x1'], 'Out')
                self.check_grad_with_place(place, ['x2'], 'Out')

    class TestConcatOpAxis0XPU(TestConcatOp):
        def init_axis(self):
            self.axis = 0

    class TestConcatOpAxis1XPU(TestConcatOp):
        def set_inputs(self):
            self.x0 = np.random.random((5, 1, 4, 5)).astype(self.dtype)
            self.x1 = np.random.random((5, 2, 4, 5)).astype(self.dtype)
            self.x2 = np.random.random((5, 3, 4, 5)).astype(self.dtype)

        def init_axis(self):
            self.axis = 1

    class TestConcatOpAxis2XPU(TestConcatOp):
        def init_axis(self):
            self.axis = 2

    class TestConcatOpAxis3XPU(TestConcatOp):
        def init_axis(self):
            self.axis = 3

    class TestConcatOpAxisNeg1XPU(TestConcatOp):
        def init_axis(self):
            self.axis = -1

    class TestConcatOpAxisNeg2XPU(TestConcatOp):
        def init_axis(self):
            self.axis = -2

    class TestConcatOpAxisNeg3XPU(TestConcatOp):
        def init_axis(self):
            self.axis = -3

    @skip_check_grad_ci(
        reason="The function 'check_grad' for large inputs is too slow.")
    class TestConcatOp3(TestConcatOp):
        def set_inputs(self):
            self.x0 = np.random.random((1, 256, 170, 256)).astype(self.dtype)
            self.x1 = np.random.random((1, 128, 170, 256)).astype(self.dtype)
            self.x2 = np.random.random((1, 128, 170, 256)).astype(self.dtype)
            self.axis = 1

        def test_check_grad(self):
            pass

    @skip_check_grad_ci(
        reason="This test will meet fetch error when there is a null grad. The detailed information is in PR#17015."
    )
    class TestConcatOp4(TestConcatOp):
        def set_inputs(self):
            self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
            self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
            self.x2 = np.random.random((0, 3, 4, 5)).astype(self.dtype)
            self.axis = 0

        def test_check_grad(self):
            pass


support_types = get_xpu_op_support_types('concat')
for stype in support_types:
    create_test_class(globals(), XPUTestConcatOp, stype)

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
