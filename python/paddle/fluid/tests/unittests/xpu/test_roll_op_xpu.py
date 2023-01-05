# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

sys.path.append("..")
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

paddle.enable_static()


class XPUTestRollOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "roll"
        self.use_dynamic_create_class = False

    class TestXPURollOp(XPUOpTest):
        def setUp(self):
            self.op_type = "roll"
            self.dtype = self.in_type
            self.init_shapes()
            self.inputs = {
                'X': np.random.random(self.x_shape).astype(self.dtype)
            }
            self.attrs = {'shifts': self.shifts, 'axis': self.axis}
            self.outputs = {
                'Out': np.roll(
                    self.inputs['X'], self.attrs['shifts'], self.attrs['axis']
                )
            }

        def init_shapes(self):
            self.x_shape = (100, 4, 5)
            self.shifts = [101, -1]
            self.axis = [0, -2]

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

        def test_check_grad(self):
            self.check_grad_with_place(paddle.XPUPlace(0), ['X'], 'Out')

    class TestRollOpCase2(TestXPURollOp):
        def init_shapes(self):
            self.x_shape = (100, 10, 5)
            self.shifts = [8, -1]
            self.axis = [-1, -2]

    class TestRollOpCase3(TestXPURollOp):
        def init_shapes(self):
            self.x_shape = (100, 10, 5, 10, 15)
            self.shifts = [50, -1, 3]
            self.axis = [-1, -2, 1]

    class TestRollOpCase4(TestXPURollOp):
        def init_shapes(self):
            self.x_shape = (100, 10, 5, 10, 15)
            self.shifts = [8, -1]
            self.axis = [-1, -2]

    class TestRollOpCase5(TestXPURollOp):
        def init_shapes(self):
            self.x_shape = (100, 10, 5, 10)
            self.shifts = [20, -1]
            self.axis = [0, -2]


support_types = get_xpu_op_support_types('roll')
for stype in support_types:
    create_test_class(globals(), XPUTestRollOp, stype)

if __name__ == "__main__":
    unittest.main()
