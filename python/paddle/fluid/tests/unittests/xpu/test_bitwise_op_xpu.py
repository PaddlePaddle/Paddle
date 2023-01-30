#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
# ----------------- TEST OP: BitwiseAnd -------------------- #
class XPUTestBitwiseAnd(XPUOpTestWrapper):
=======
################## TEST OP: BitwiseAnd ##################
class XPUTestBitwiseAnd(XPUOpTestWrapper):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'bitwise_and'

    class XPUTestBitwiseAndBase(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'bitwise_and'

<<<<<<< HEAD
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
            out = np.bitwise_and(x, y)

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
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestBitwiseAndCase1(XPUTestBitwiseAndBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseAndCase2(XPUTestBitwiseAndBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [4, 1]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseAndCase3(XPUTestBitwiseAndBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = 0
            self.high = 100


support_types = get_xpu_op_support_types('bitwise_and')
for stype in support_types:
    create_test_class(globals(), XPUTestBitwiseAnd, stype)


<<<<<<< HEAD
# -------------- TEST OP: BitwiseOr ----------------- #
class XPUTestBitwiseOr(XPUOpTestWrapper):
=======
################## TEST OP: BitwiseOr ##################
class XPUTestBitwiseOr(XPUOpTestWrapper):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'bitwise_or'

    class XPUTestBitwiseOrBase(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'bitwise_or'

<<<<<<< HEAD
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
            out = np.bitwise_or(x, y)

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
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestBitwiseOrCase1(XPUTestBitwiseOrBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseOrCase2(XPUTestBitwiseOrBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [4, 1]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseOrCase3(XPUTestBitwiseOrBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = 0
            self.high = 100


support_types = get_xpu_op_support_types('bitwise_or')
for stype in support_types:
    create_test_class(globals(), XPUTestBitwiseOr, stype)


<<<<<<< HEAD
# --------------- TEST OP: BitwiseXor ---------------- #
class XPUTestBitwiseXor(XPUOpTestWrapper):
=======
################## TEST OP: BitwiseXor ##################
class XPUTestBitwiseXor(XPUOpTestWrapper):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'bitwise_xor'

    class XPUTestBitwiseXorBase(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'bitwise_xor'

<<<<<<< HEAD
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
            out = np.bitwise_xor(x, y)

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
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            pass

    class XPUTestBitwiseXorCase1(XPUTestBitwiseXorBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseXorCase2(XPUTestBitwiseXorBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [4, 1]
            self.low = -100
            self.high = 100

    class XPUTestBitwiseXorCase3(XPUTestBitwiseXorBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_case(self):
            self.dtype = np.int32
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]
            self.low = 0
            self.high = 100


support_types = get_xpu_op_support_types('bitwise_xor')
for stype in support_types:
    create_test_class(globals(), XPUTestBitwiseXor, stype)


<<<<<<< HEAD
# ----------------  TEST OP: BitwiseNot ------------------ #
class XPUTestBitwiseNot(XPUOpTestWrapper):
=======
##################  TEST OP: BitwiseNot ##################
class XPUTestBitwiseNot(XPUOpTestWrapper):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'bitwise_not'

    class XPUTestBitwiseNotBase(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'bitwise_not'

<<<<<<< HEAD
            x = np.random.randint(
                self.low, self.high, self.x_shape, dtype=self.dtype
            )
=======
            x = np.random.randint(self.low,
                                  self.high,
                                  self.x_shape,
                                  dtype=self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out = np.bitwise_not(x)

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

    class XPUTestBitwiseNotBool(XPUTestBitwiseNotBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_case()
            self.set_case()

        def set_case(self):
            self.op_type = 'bitwise_not'

            x = np.random.choice([True, False], self.x_shape)
            out = np.bitwise_not(x)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': x}
            self.outputs = {'Out': out}

        def init_case(self):
<<<<<<< HEAD
            self.dtype = np.bool_
=======
            self.dtype = np.bool
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.x_shape = [2, 3, 4, 5]


support_types = get_xpu_op_support_types('bitwise_not')
for stype in support_types:
    create_test_class(globals(), XPUTestBitwiseNot, stype)

if __name__ == '__main__':
    unittest.main()
