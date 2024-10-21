#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


def l2_norm(x, axis, epsilon):
    x2 = x**2
    s = np.sum(x2, axis=axis, keepdims=True)
    r = np.sqrt(s + epsilon)
    y = x / np.broadcast_to(r, x.shape)
    return y, r


class XPUTestNormOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "norm"
        self.use_dynamic_create_class = False

    class TestXPUNormOp(XPUOpTest):
        def setUp(self):
            self.op_type = "norm"
            self.dtype = (
                self.in_type if self.in_type != np.uint16 else np.float32
            )
            self.place = paddle.XPUPlace(0)
            self.init_test_case()
            x = np.random.random(self.shape).astype(self.dtype)
            y, norm = l2_norm(x, self.axis, self.epsilon)
            if self.in_type == np.uint16:
                x = convert_float_to_uint16(x)
            self.inputs = {'X': x}
            self.attrs = {'epsilon': self.epsilon, 'axis': self.axis}
            self.outputs = {'Out': y, 'Norm': norm}

        def init_test_case(self):
            self.shape = [2, 3, 4, 5]
            self.axis = 1
            self.epsilon = 1e-8

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestXPUNormOp2(TestXPUNormOp):
        def init_test_case(self):
            self.shape = [5, 3, 9, 7]
            self.axis = 0
            self.epsilon = 1e-8

    class TestXPUNormOp3(TestXPUNormOp):
        def init_test_case(self):
            self.shape = [5, 3, 2, 7]
            self.axis = -1
            self.epsilon = 1e-8

    class TestXPUNormOp4(TestXPUNormOp):
        def init_test_case(self):
            self.shape = [128, 1024, 14, 14]
            self.axis = 2
            self.epsilon = 1e-8

    class TestXPUNormOp5(TestXPUNormOp):
        def init_test_case(self):
            self.shape = [2048, 2048]
            self.axis = 1
            self.epsilon = 1e-8


support_types = get_xpu_op_support_types('norm')
for stype in support_types:
    create_test_class(globals(), XPUTestNormOp, stype)

if __name__ == "__main__":
    unittest.main()
