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

import sys
import unittest

import numpy as np

sys.path.append("..")

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle

paddle.enable_static()


class XPUTestAssignOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'assign'
        self.use_dynamic_create_class = False

    class TestAssignOPBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.set_case()

        def set_case(self):
            self.op_type = 'assign'
            self.init_config()

            x = np.random.random(size=self.input_shape).astype(self.dtype)
            self.inputs = {'X': x}
            self.attrs = {}
            self.outputs = {'Out': x}

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def init_config(self):
            self.input_shape = (2, 5)

    class XPUTestAssign1(TestAssignOPBase):
        def init_config(self):
            self.input_shape = [2, 768]

    class XPUTestAssign2(TestAssignOPBase):
        def init_config(self):
            self.input_shape = [3, 8, 4096]

    class XPUTestAssign3(TestAssignOPBase):
        def init_config(self):
            self.input_shape = [1024]

    class XPUTestAssign4(TestAssignOPBase):
        def init_config(self):
            self.input_shape = [2, 2, 255]


support_types = get_xpu_op_support_types('assign')
for stype in support_types:
    create_test_class(globals(), XPUTestAssignOP, stype)

if __name__ == "__main__":
    unittest.main()
