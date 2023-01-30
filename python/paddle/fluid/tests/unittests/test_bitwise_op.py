#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
<<<<<<< HEAD

import numpy as np
from op_test import OpTest

import paddle
=======
import numpy as np
import paddle
from op_test import OpTest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


<<<<<<< HEAD
# ----------------- TEST OP: BitwiseAnd ----------------- #
class TestBitwiseAnd(OpTest):
=======
################## TEST OP: BitwiseAnd ##################
class TestBitwiseAnd(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "bitwise_and"
        self.init_dtype()
        self.init_shape()
        self.init_bound()

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

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


<<<<<<< HEAD
class TestBitwiseAnd_ZeroDim1(TestBitwiseAnd):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestBitwiseAnd_ZeroDim2(TestBitwiseAnd):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []


class TestBitwiseAnd_ZeroDim3(TestBitwiseAnd):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseAndUInt8(TestBitwiseAnd):
=======
class TestBitwiseAndUInt8(TestBitwiseAnd):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseAndInt8(TestBitwiseAnd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseAndInt16(TestBitwiseAnd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


class TestBitwiseAndInt64(TestBitwiseAnd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseAndBool(TestBitwiseAnd):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "bitwise_and"
        self.init_shape()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_and(x, y)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}


<<<<<<< HEAD
# ----------------- TEST OP: BitwiseOr ------------------ #
class TestBitwiseOr(OpTest):
=======
################## TEST OP: BitwiseOr ##################
class TestBitwiseOr(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "bitwise_or"
        self.init_dtype()
        self.init_shape()
        self.init_bound()

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

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


<<<<<<< HEAD
class TestBitwiseOr_ZeroDim1(TestBitwiseOr):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestBitwiseOr_ZeroDim2(TestBitwiseOr):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []


class TestBitwiseOr_ZeroDim3(TestBitwiseOr):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseOrUInt8(TestBitwiseOr):
=======
class TestBitwiseOrUInt8(TestBitwiseOr):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseOrInt8(TestBitwiseOr):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseOrInt16(TestBitwiseOr):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


class TestBitwiseOrInt64(TestBitwiseOr):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseOrBool(TestBitwiseOr):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "bitwise_or"
        self.init_shape()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_or(x, y)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}


<<<<<<< HEAD
# ----------------- TEST OP: BitwiseXor ---------------- #
class TestBitwiseXor(OpTest):
=======
################## TEST OP: BitwiseXor ##################
class TestBitwiseXor(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "bitwise_xor"
        self.init_dtype()
        self.init_shape()
        self.init_bound()

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

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


<<<<<<< HEAD
class TestBitwiseXor_ZeroDim1(TestBitwiseXor):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestBitwiseXor_ZeroDim2(TestBitwiseXor):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []


class TestBitwiseXor_ZeroDim3(TestBitwiseXor):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseXorUInt8(TestBitwiseXor):
=======
class TestBitwiseXorUInt8(TestBitwiseXor):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseXorInt8(TestBitwiseXor):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseXorInt16(TestBitwiseXor):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


class TestBitwiseXorInt64(TestBitwiseXor):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseXorBool(TestBitwiseXor):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "bitwise_xor"
        self.init_shape()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_xor(x, y)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}


<<<<<<< HEAD
# ---------------  TEST OP: BitwiseNot ----------------- #
class TestBitwiseNot(OpTest):
=======
##################  TEST OP: BitwiseNot ##################
class TestBitwiseNot(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "bitwise_not"
        self.init_dtype()
        self.init_shape()
        self.init_bound()

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

        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


<<<<<<< HEAD
class TestBitwiseNot_ZeroDim(TestBitwiseNot):
    def init_shape(self):
        self.x_shape = []


class TestBitwiseNotUInt8(TestBitwiseNot):
=======
class TestBitwiseNotUInt8(TestBitwiseNot):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseNotInt8(TestBitwiseNot):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]


class TestBitwiseNotInt16(TestBitwiseNot):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
<<<<<<< HEAD


class TestBitwiseNotInt64(TestBitwiseNot):
=======
        self.y_shape = [4, 1]


class TestBitwiseNotInt64(TestBitwiseNot):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]


class TestBitwiseNotBool(TestBitwiseNot):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "bitwise_not"
        self.init_shape()

        x = np.random.choice([True, False], self.x_shape)
        out = np.bitwise_not(x)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}


if __name__ == "__main__":
    unittest.main()
