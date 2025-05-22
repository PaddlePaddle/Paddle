# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


def ref_complex(x, y):
    return x + 1j * y


class XPUTestComplexOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'complex'
        self.use_dynamic_create_class = False

    class TestComplexOp(XPUOpTest):
        def setUp(self):
            self.op_type = "complex"
            self.python_api = paddle.complex
            self.init_spec()
            self.init_type()
            x = np.random.randn(*self.x_shape).astype(self.dtype)
            y = np.random.randn(*self.y_shape).astype(self.dtype)
            out_ref = ref_complex(x, y)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {'Out': out_ref}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X', 'Y'], 'Out')

        def test_check_grad_ignore_x(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place, ['Y'], 'Out', no_grad_set=set("X")
                )

        def test_check_grad_ignore_y(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place, ['X'], 'Out', no_grad_set=set("Y")
                )

        def init_type(self):
            self.dtype = self.in_type

        def init_spec(self):
            self.x_shape = [10, 10]
            self.y_shape = [10, 10]

    class TestComplexOpBroadcast1(TestComplexOp):
        def init_spec(self):
            self.x_shape = [10, 3, 1, 4]
            self.y_shape = [100, 1]

    class TestComplexOpBroadcast2(TestComplexOp):
        def init_spec(self):
            self.x_shape = [100, 1]
            self.y_shape = [10, 3, 1, 4]

    class TestComplexOpBroadcast3(TestComplexOp):
        def init_spec(self):
            self.x_shape = [1, 100]
            self.y_shape = [100]

    class TestComplexOpZeroSize1(TestComplexOp):
        def init_spec(self):
            self.x_shape = [0, 3, 1, 4]
            self.y_shape = [1, 1]

    class TestComplexOpZeroSize2(TestComplexOp):
        def init_spec(self):
            self.x_shape = [10, 0]
            self.y_shape = [10, 0]

    class TestComplexOpZeroSize3(TestComplexOp):
        def init_spec(self):
            self.x_shape = [3, 0, 1, 4]
            self.y_shape = [1, 1]

    class TestComplexOpZeroSize4(TestComplexOp):
        def init_spec(self):
            self.x_shape = [1, 1]
            self.y_shape = [10, 3, 0, 4]


support_types = get_xpu_op_support_types('complex')
for stype in support_types:
    create_test_class(globals(), XPUTestComplexOp, stype)

if __name__ == "__main__":
    unittest.main()
