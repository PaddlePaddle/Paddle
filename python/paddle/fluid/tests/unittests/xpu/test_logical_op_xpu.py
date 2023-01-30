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

<<<<<<< HEAD
import sys
import unittest

import numpy as np

sys.path.append("..")

from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
=======
from __future__ import print_function

import unittest
import numpy as np
import sys

sys.path.append("..")

import paddle
from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


<<<<<<< HEAD
# -------------- TEST OP: logical_and ----------------- #
class XPUTestLogicalAnd(XPUOpTestWrapper):
=======
################## TEST OP: logical_and ##################
class XPUTestLogicalAnd(XPUOpTestWrapper):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'logical_and'

    class XPUTestLogicalAndBase(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'logical_and'

<<<<<<< HEAD
            # special range for bool dtype
            if self.dtype == np.dtype(np.bool_):
                self.low = 0
                self.high = 2

            x = np.random.randint(
                self.low, self.high, self.x_shape, dtype=self.dtype
            )
            y = np.random.randint(
                self.low, self.high, self.y_shape, dtype=self.dtype
            )
=======
            x = np.random.randint(self.low,
                                  self.high,
                                  self.x_shape,
                                  dtype=self.dtype)
            y = np.random.randint(self.low,
                                  self.high,
                                  self.y_shape,
                                  dtype=self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out = np.logical_and(x, y)

            self.attrs = {'use_xpu': True}
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(x),
<<<<<<< HEAD
                'Y': OpTest.np_dtype_to_fluid_dtype(y),
=======
                'Y': OpTest.np_dtype_to_fluid_dtype(y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {'Out': out}

        def init_case(self):
<<<<<<< HEAD
            self.dtype = self.in_type
=======
            self.dtype = np.int32
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestLogicalAndCase1(XPUTestLogicalAndBase):
<<<<<<< HEAD
        def init_case(self):
            self.dtype = self.in_type
=======

        def init_case(self):
            self.dtype = np.int32
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100


support_types = get_xpu_op_support_types('logical_and')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalAnd, stype)


<<<<<<< HEAD
# --------------- TEST OP: logical_or ------------------ #
class XPUTestLogicalOr(XPUOpTestWrapper):
=======
################## TEST OP: logical_or ##################
class XPUTestLogicalOr(XPUOpTestWrapper):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'logical_or'

    class XPUTestLogicalOrBase(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'logical_or'

<<<<<<< HEAD
            # special range for bool dtype
            if self.dtype == np.dtype(np.bool_):
                self.low = 0
                self.high = 2

            x = np.random.randint(
                self.low, self.high, self.x_shape, dtype=self.dtype
            )
            y = np.random.randint(
                self.low, self.high, self.y_shape, dtype=self.dtype
            )
=======
            x = np.random.randint(self.low,
                                  self.high,
                                  self.x_shape,
                                  dtype=self.dtype)
            y = np.random.randint(self.low,
                                  self.high,
                                  self.y_shape,
                                  dtype=self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out = np.logical_or(x, y)

            self.attrs = {'use_xpu': True}
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(x),
<<<<<<< HEAD
                'Y': OpTest.np_dtype_to_fluid_dtype(y),
=======
                'Y': OpTest.np_dtype_to_fluid_dtype(y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {'Out': out}

        def init_case(self):
<<<<<<< HEAD
            self.dtype = self.in_type
=======
            self.dtype = np.int32
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestLogicalOrCase1(XPUTestLogicalOrBase):
<<<<<<< HEAD
        def init_case(self):
            self.dtype = self.in_type
=======

        def init_case(self):
            self.dtype = np.int32
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100


support_types = get_xpu_op_support_types('logical_or')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalOr, stype)


<<<<<<< HEAD
# --------------- TEST OP: logical_xor ------------------- #
class XPUTestLogicalXor(XPUOpTestWrapper):
=======
################## TEST OP: logical_xor ##################
class XPUTestLogicalXor(XPUOpTestWrapper):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'logical_xor'

    class XPUTestLogicalXorBase(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'logical_xor'

<<<<<<< HEAD
            # special range for bool dtype
            if self.dtype == np.dtype(np.bool_):
                self.low = 0
                self.high = 2

            x = np.random.randint(
                self.low, self.high, self.x_shape, dtype=self.dtype
            )
            y = np.random.randint(
                self.low, self.high, self.y_shape, dtype=self.dtype
            )
=======
            x = np.random.randint(self.low,
                                  self.high,
                                  self.x_shape,
                                  dtype=self.dtype)
            y = np.random.randint(self.low,
                                  self.high,
                                  self.y_shape,
                                  dtype=self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out = np.logical_xor(x, y)

            self.attrs = {'use_xpu': True}
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(x),
<<<<<<< HEAD
                'Y': OpTest.np_dtype_to_fluid_dtype(y),
=======
                'Y': OpTest.np_dtype_to_fluid_dtype(y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {'Out': out}

        def init_case(self):
<<<<<<< HEAD
            self.dtype = self.in_type
=======
            self.dtype = np.int64
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestLogicalXorCase1(XPUTestLogicalXorBase):
<<<<<<< HEAD
        def init_case(self):
            self.dtype = self.in_type
=======

        def init_case(self):
            self.dtype = np.int32
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100


support_types = get_xpu_op_support_types('logical_xor')
for stype in support_types:
    create_test_class(globals(), XPUTestLogicalXor, stype)


<<<<<<< HEAD
# -------------  TEST OP: LogicalNot ---------------- #
class XPUTestLogicalNot(XPUOpTestWrapper):
=======
##################  TEST OP: LogicalNot ##################
class XPUTestLogicalNot(XPUOpTestWrapper):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'logical_not'

    class XPUTestLogicalNotBase(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'logical_not'

<<<<<<< HEAD
            # special range for bool dtype
            if self.dtype == np.dtype(np.bool_):
                self.low = 0
                self.high = 2

            x = np.random.randint(
                self.low, self.high, self.x_shape, dtype=self.dtype
            )
=======
            x = np.random.randint(self.low,
                                  self.high,
                                  self.x_shape,
                                  dtype=self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out = np.logical_not(x)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': out}

        def init_case(self):
<<<<<<< HEAD
            self.dtype = self.in_type
=======
            self.dtype = np.int32
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
