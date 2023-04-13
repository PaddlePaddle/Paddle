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

import paddle

paddle.enable_static()


from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest


class XPUTestAtanOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "atan"
        self.use_dynamic_create_class = False

    class TestAtanOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "atan"

            # override
            self.init_input_shape()

            x = np.random.random(self.x_shape).astype(self.in_type)
            y = np.arctan(x)

            self.inputs = {'X': x}
            self.outputs = {'Out': y}

        def init_input_shape(self):
            self.x_shape = [15, 6]

        def set_xpu(self):
            self.__class__.no_need_check_grad = False
            self.place = paddle.XPUPlace(0)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class Test1x1(TestAtanOp):
        def init_input_shape(self):
            self.x_shape = [1, 1]

    class Test1(TestAtanOp):
        def init_input_shape(self):
            self.x_shape = [1]


support_types = get_xpu_op_support_types("atan")
for stype in support_types:
    create_test_class(globals(), XPUTestAtanOp, stype)

if __name__ == "__main__":
    unittest.main()
