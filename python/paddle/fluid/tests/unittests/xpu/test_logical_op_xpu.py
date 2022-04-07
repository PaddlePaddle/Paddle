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

import unittest
import numpy as np
import sys
sys.path.append("..")

import paddle
from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


################## TEST OP: logical_and ##################
class XPUTestLogicalAnd(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'logical_and'

    class XPUTestLogicalAndBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'logical_and'

            x = np.random.randint(
                self.low, self.high, self.x_shape, dtype=self.dtype)
            y = np.random.randint(
                self.low, self.high, self.y_shape, dtype=self.dtype)
            out = np.logical_and(x, y)

            self.attrs = {'use_xpu': True}
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(x),
                'Y': OpTest.np_dtype_to_fluid_dtype(y)
            }
            self.outputs = {'Out': out}

        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestLogicalAndCase1(XPUTestLogicalAndBase):
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100


support_types = get_xpu_op_support_types('logical_and')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalAnd, stype)


################## TEST OP: logical_or ##################
class XPUTestLogicalOr(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'logical_or'

    class XPUTestLogicalOrBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'logical_or'

            x = np.random.randint(
                self.low, self.high, self.x_shape, dtype=self.dtype)
            y = np.random.randint(
                self.low, self.high, self.y_shape, dtype=self.dtype)
            out = np.logical_or(x, y)

            self.attrs = {'use_xpu': True}
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(x),
                'Y': OpTest.np_dtype_to_fluid_dtype(y)
            }
            self.outputs = {'Out': out}

        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestLogicalOrCase1(XPUTestLogicalOrBase):
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100


support_types = get_xpu_op_support_types('logical_or')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalOr, stype)


################## TEST OP: logical_xor ##################
class XPUTestLogicalXor(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'logical_xor'

    class XPUTestLogicalXorBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'logical_xor'

            x = np.random.randint(
                self.low, self.high, self.x_shape, dtype=self.dtype)
            y = np.random.randint(
                self.low, self.high, self.y_shape, dtype=self.dtype)
            out = np.logical_xor(x, y)

            self.attrs = {'use_xpu': True}
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(x),
                'Y': OpTest.np_dtype_to_fluid_dtype(y)
            }
            self.outputs = {'Out': out}

        def init_case(self):
            self.dtype = np.int64
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestLogicalXorCase1(XPUTestLogicalXorBase):
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100


support_types = get_xpu_op_support_types('logical_xor')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalXor, stype)


##################  TEST OP: LogicalNot ##################
class XPUTestLogicalNot(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'logical_not'

    class XPUTestLogicalNotBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'logical_not'

            x = np.random.randint(
                self.low, self.high, self.x_shape, dtype=self.dtype)
            out = np.logical_not(x)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': out}

        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass


support_types = get_xpu_op_support_types('logical_not')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalNot, stype)

if __name__ == '__main__':
    unittest.main()
