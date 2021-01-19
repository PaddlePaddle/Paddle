#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# Test set_value op in static mode

from __future__ import print_function

import unittest
import numpy as np

import paddle


class TestSetValueBase(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.set_value()
        self.shape = [2, 3, 4]
        self.data = np.ones(self.shape).astype(self.dtype)
        self.program = paddle.static.Program()

    def set_value(self):
        self.value = 6

    def set_dtype(self):
        self.dtype = "float32"

    def _call_setitem(self, x):
        x[0, 0] = self.value

    def _get_answer(self):
        self.data[0, 0] = self.value


class TestSetValueApi(TestSetValueBase):
    def test_api(self):
        with paddle.static.program_guard(self.program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            self._call_setitem(x)

        exe = paddle.static.Executor(paddle.CPUPlace())
        out = exe.run(self.program, fetch_list=[x])

        self._get_answer()
        self.assertTrue(
            (self.data == out).all(),
            msg="\nExpected res = \n{}, \n\nbut received : \n{}".format(
                self.data, out))


# 1. Test different type of item: int, python slice
class TestSetValueItemInt(TestSetValueApi):
    def _call_setitem(self, x):
        x[0] = self.value

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueItemSlice(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:2] = self.value

    def _get_answer(self):
        self.data[0:2] = self.value


class TestSetValueItemSlice2(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1] = self.value

    def _get_answer(self):
        self.data[0:-1] = self.value


class TestSetValueItemSlice3(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1, 0:2] = self.value

    def _get_answer(self):
        self.data[0:-1, 0:2] = self.value


class TestSetValueItemSlice4(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:2, :] = self.value

    def _get_answer(self):
        self.data[0:, 1:2, :] = self.value


# 2. Test different type of value: int, float, numpy.ndarray, Tensor
# 2.1 value is int32, int64, float32, float64, bool


def create_test_value_int32(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 7

        def set_dtype(self):
            self.dtype = "int32"

    cls_name = "{0}_{1}".format(parent.__name__, "ValueInt32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_int32(TestSetValueItemInt)
create_test_value_int32(TestSetValueItemSlice)
create_test_value_int32(TestSetValueItemSlice2)
create_test_value_int32(TestSetValueItemSlice3)
create_test_value_int32(TestSetValueItemSlice4)


def create_test_value_int64(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 7

        def set_dtype(self):
            self.dtype = "int64"

    cls_name = "{0}_{1}".format(parent.__name__, "ValueInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_int64(TestSetValueItemInt)
create_test_value_int64(TestSetValueItemSlice)
create_test_value_int64(TestSetValueItemSlice2)
create_test_value_int64(TestSetValueItemSlice3)
create_test_value_int64(TestSetValueItemSlice4)


def create_test_value_fp32(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 3.3

        def set_dtype(self):
            self.dtype = "float32"

    cls_name = "{0}_{1}".format(parent.__name__, "ValueFp32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_fp32(TestSetValueItemInt)
create_test_value_fp32(TestSetValueItemSlice)
create_test_value_fp32(TestSetValueItemSlice2)
create_test_value_fp32(TestSetValueItemSlice3)
create_test_value_fp32(TestSetValueItemSlice4)


def create_test_value_fp64(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 2.0**127  # float32:[-2^128, 2^128)

        def set_dtype(self):
            self.dtype = "float64"

    cls_name = "{0}_{1}".format(parent.__name__, "ValueFp64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_fp64(TestSetValueItemInt)
create_test_value_fp64(TestSetValueItemSlice)
create_test_value_fp64(TestSetValueItemSlice2)
create_test_value_fp64(TestSetValueItemSlice3)
create_test_value_fp64(TestSetValueItemSlice4)


def create_test_value_bool(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 0

        def set_dtype(self):
            self.dtype = "bool"

    cls_name = "{0}_{1}".format(parent.__name__, "ValueBool")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_bool(TestSetValueItemInt)
create_test_value_bool(TestSetValueItemSlice)
create_test_value_bool(TestSetValueItemSlice2)
create_test_value_bool(TestSetValueItemSlice3)
create_test_value_bool(TestSetValueItemSlice4)


# 2.2 value is numpy.array (int32, int64, float32, float64, bool)
def create_test_value_numpy_int32(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = np.array([5])

        def set_dtype(self):
            self.dtype = "int32"

    cls_name = "{0}_{1}".format(parent.__name__, "ValueNumpyInt32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_int32(TestSetValueItemInt)
create_test_value_numpy_int32(TestSetValueItemSlice)
create_test_value_numpy_int32(TestSetValueItemSlice2)
create_test_value_numpy_int32(TestSetValueItemSlice3)
create_test_value_numpy_int32(TestSetValueItemSlice4)


def create_test_value_numpy_int64(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = np.array([1])

        def set_dtype(self):
            self.dtype = "int64"

    cls_name = "{0}_{1}".format(parent.__name__, "ValueNumpyInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_int64(TestSetValueItemInt)
create_test_value_numpy_int64(TestSetValueItemSlice)
create_test_value_numpy_int64(TestSetValueItemSlice2)
create_test_value_numpy_int64(TestSetValueItemSlice3)
create_test_value_numpy_int64(TestSetValueItemSlice4)


def create_test_value_numpy_fp32(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = np.array([1])

        def set_dtype(self):
            self.dtype = "float32"

    cls_name = "{0}_{1}".format(parent.__name__, "ValueNumpyFp32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_fp32(TestSetValueItemInt)
create_test_value_numpy_fp32(TestSetValueItemSlice)
create_test_value_numpy_fp32(TestSetValueItemSlice2)
create_test_value_numpy_fp32(TestSetValueItemSlice3)
create_test_value_numpy_fp32(TestSetValueItemSlice4)


def create_test_value_numpy_fp64(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = np.array([2**127]).astype("float64")

        def set_dtype(self):
            self.dtype = "float64"

    cls_name = "{0}_{1}".format(parent.__name__, "ValueNumpyFp64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_fp64(TestSetValueItemInt)
create_test_value_numpy_fp64(TestSetValueItemSlice)
create_test_value_numpy_fp64(TestSetValueItemSlice2)
create_test_value_numpy_fp64(TestSetValueItemSlice3)
create_test_value_numpy_fp64(TestSetValueItemSlice4)


def create_test_value_numpy_bool(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = np.array([0])

        def set_dtype(self):
            self.dtype = "bool"

    cls_name = "{0}_{1}".format(parent.__name__, "ValueNumpyBool")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_bool(TestSetValueItemInt)
create_test_value_numpy_bool(TestSetValueItemSlice)
create_test_value_numpy_bool(TestSetValueItemSlice2)
create_test_value_numpy_bool(TestSetValueItemSlice3)
create_test_value_numpy_bool(TestSetValueItemSlice4)


# 2.3 value is a Paddle Tensor (int32, int64, float32, float64, bool)
def create_test_value_tensor_int32(parent):
    class TestValueInt(parent):
        def set_dtype(self):
            self.dtype = "int32"

        def _call_setitem(self, x):
            value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{0}_{1}".format(parent.__name__, "ValueTensorInt32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_int32(TestSetValueItemInt)
create_test_value_tensor_int32(TestSetValueItemSlice)
create_test_value_tensor_int32(TestSetValueItemSlice2)
create_test_value_tensor_int32(TestSetValueItemSlice3)
create_test_value_tensor_int32(TestSetValueItemSlice4)


def create_test_value_tensor_int64(parent):
    class TestValueInt(parent):
        def set_dtype(self):
            self.dtype = "int64"

        def _call_setitem(self, x):
            value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{0}_{1}".format(parent.__name__, "ValueTensorInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_int64(TestSetValueItemInt)
create_test_value_tensor_int64(TestSetValueItemSlice)
create_test_value_tensor_int64(TestSetValueItemSlice2)
create_test_value_tensor_int64(TestSetValueItemSlice3)
create_test_value_tensor_int64(TestSetValueItemSlice4)


def create_test_value_tensor_fp32(parent):
    class TestValueInt(parent):
        def set_dtype(self):
            self.dtype = "float32"

        def _call_setitem(self, x):
            value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{0}_{1}".format(parent.__name__, "ValueTensorFp32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_fp32(TestSetValueItemInt)
create_test_value_tensor_fp32(TestSetValueItemSlice)
create_test_value_tensor_fp32(TestSetValueItemSlice2)
create_test_value_tensor_fp32(TestSetValueItemSlice3)
create_test_value_tensor_fp32(TestSetValueItemSlice4)


def create_test_value_tensor_fp64(parent):
    class TestValueInt(parent):
        def set_dtype(self):
            self.dtype = "float64"

        def _call_setitem(self, x):
            value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{0}_{1}".format(parent.__name__, "ValueTensorFp64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_fp64(TestSetValueItemInt)
create_test_value_tensor_fp64(TestSetValueItemSlice)
create_test_value_tensor_fp64(TestSetValueItemSlice2)
create_test_value_tensor_fp64(TestSetValueItemSlice3)
create_test_value_tensor_fp64(TestSetValueItemSlice4)


def create_test_value_tensor_bool(parent):
    class TestValueInt(parent):
        def set_dtype(self):
            self.dtype = "bool"

        def _call_setitem(self, x):
            value = paddle.full(shape=[1], fill_value=False, dtype=self.dtype)
            x[0, 1] = value

        def _get_answer(self):
            self.data[0, 1] = False

    cls_name = "{0}_{1}".format(parent.__name__, "ValueTensorBool")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_bool(TestSetValueItemInt)
create_test_value_tensor_bool(TestSetValueItemSlice)
create_test_value_tensor_bool(TestSetValueItemSlice2)
create_test_value_tensor_bool(TestSetValueItemSlice3)
create_test_value_tensor_bool(TestSetValueItemSlice4)


# 3. Test different shape of value
class TestSetValueValueShape1(TestSetValueApi):
    def set_value(self):
        self.value = np.array([3, 4, 5, 6])  # shape is (4,)

    def _call_setitem(self, x):
        x[0] = self.value

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueValueShape2(TestSetValueApi):
    def set_value(self):
        self.value = np.array([[3, 4, 5, 6]])  # shape is (1,4)

    def _call_setitem(self, x):
        x[0:1] = self.value

    def _get_answer(self):
        self.data[0:1] = self.value


class TestSetValueValueShape3(TestSetValueApi):
    def set_value(self):
        self.value = np.array(
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])  # shape is (3,4)

    def _call_setitem(self, x):
        x[0] = self.value

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueValueShape4(TestSetValueApi):
    def set_value(self):
        self.value = np.array(
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]).astype(
                self.dtype)  # shape is (3,4)

    def _call_setitem(self, x):
        x[0] = paddle.assign(self.value)  # x is Paddle.Tensor

    def _get_answer(self):
        self.data[0] = self.value


# 4. Test error
class TestError(TestSetValueBase):
    def _value_type_error(self):
        with self.assertRaisesRegexp(
                TypeError,
                "Only support to assign an integer, float, numpy.ndarray or paddle.Tensor"
        ):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            value = [1]
            x[0] = value

    def _dtype_error(self):
        with self.assertRaisesRegexp(
                TypeError,
                "When assign a numpy.ndarray, integer or float to a paddle.Tensor, "
        ):
            y = paddle.ones(shape=self.shape, dtype="float16")
            y[0] = 1

    def _step_error(self):
        with self.assertRaisesRegexp(ValueError, "only support step is 1"):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            x[0:1:2] = self.value

    def _broadcast_mismatch(self):
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            value = np.array([3, 4, 5, 6, 7])
            x[0] = value
        exe = paddle.static.Executor(paddle.CPUPlace())
        with self.assertRaisesRegexp(ValueError,
                                     "Broadcast dimension mismatch."):
            exe.run(program)

    def test_error(self):
        with paddle.static.program_guard(self.program):
            self._value_type_error()
            self._dtype_error()
            self._step_error()
        self._broadcast_mismatch()


if __name__ == '__main__':
    unittest.main()
