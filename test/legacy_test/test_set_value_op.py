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

# Test set_value op in static graph mode

import os
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core


class TestSetValueBase(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.set_value()
        self.set_shape()
        self.data = np.ones(self.shape).astype(self.dtype)
        self.program = paddle.static.Program()

    def set_shape(self):
        self.shape = [2, 3, 4]

    def set_value(self):
        self.value = 6

    def set_dtype(self):
        self.dtype = "float32"

    def _call_setitem(self, x):
        x[0, 0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (0, 0), self.value)
        return x

    def _get_answer(self):
        self.data[0, 0] = self.value


class TestSetValueApi(TestSetValueBase):
    def _run_static(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            x = self._call_setitem_static_api(x)

        exe = paddle.static.Executor(paddle.CPUPlace())
        out = exe.run(main_program, fetch_list=[x])
        paddle.disable_static()
        return out

    def _run_dynamic(self):
        paddle.disable_static()
        x = paddle.ones(shape=self.shape, dtype=self.dtype)
        self._call_setitem(x)
        out = x.numpy()
        paddle.enable_static()
        return out

    def test_api(self):
        static_out = self._run_static()
        dynamic_out = self._run_dynamic()
        self._get_answer()

        error_msg = (
            "\nIn {} mode: \nExpected res = \n{}, \n\nbut received : \n{}"
        )
        self.assertTrue(
            (self.data == static_out).all(),
            msg=error_msg.format("static", self.data, static_out),
        )
        self.assertTrue(
            (self.data == dynamic_out).all(),
            msg=error_msg.format("dynamic", self.data, dynamic_out),
        )


# 1. Test different type of item: int, Python slice, Paddle Tensor
# 1.1 item is int
class TestSetValueItemInt(TestSetValueApi):
    def _call_setitem(self, x):
        x[0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, 0, self.value)
        return x

    def _get_answer(self):
        self.data[0] = self.value


# 1.2 item is slice
# 1.2.1 step is 1
class TestSetValueItemSlice(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:2] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(0, 2), self.value)
        return x

    def _get_answer(self):
        self.data[0:2] = self.value


class TestSetValueItemSlice2(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(0, -1), self.value)
        return x

    def _get_answer(self):
        self.data[0:-1] = self.value


class TestSetValueItemSlice3(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1, 0:2] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (slice(0, -1), slice(0, 2)), self.value)
        return x

    def _get_answer(self):
        self.data[0:-1, 0:2] = self.value


class TestSetValueItemSlice4(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:2, :] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(0, None), slice(1, 2), slice(None)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:, 1:2, :] = self.value


class TestSetValueItemSlice5(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:1, :] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(0, None), slice(1, 1), slice(None)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:, 1:1, :] = self.value


class TestSetValueItemSliceInWhile(TestSetValueApi):
    def _call_setitem(self, x):
        def cond(i, x):
            return i < 1

        def body(i, x):
            x[i] = self.value
            i = i + 1
            return i, x

        i = paddle.zeros(shape=(1,), dtype='int32')
        i, x = paddle.static.nn.while_loop(cond, body, [i, x])

    def _call_setitem_static_api(self, x):
        def cond(i, x):
            return i < 1

        def body(i, x):
            x = paddle.static.setitem(x, i, self.value)
            i = i + 1
            return i, x

        i = paddle.zeros(shape=(1,), dtype='int32')
        i, x = paddle.static.nn.while_loop(cond, body, [i, x])
        return x

    def _get_answer(self):
        self.data[0] = self.value


# 1.2.2 step > 1
class TestSetValueItemSliceStep(TestSetValueApi):
    def set_shape(self):
        self.shape = [5, 5, 5]

    def _call_setitem(self, x):
        x[0:2:2] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(0, 2, 2), self.value)
        return x

    def _get_answer(self):
        self.data[0:2:2] = self.value


class TestSetValueItemSliceStep2(TestSetValueApi):
    def set_shape(self):
        self.shape = [7, 5, 5]

    def _call_setitem(self, x):
        x[0:-1:3] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(0, -1, 3), self.value)
        return x

    def _get_answer(self):
        self.data[0:-1:3] = self.value


class TestSetValueItemSliceStep3(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1, 0:2, ::2] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(0, -1), slice(0, 2), slice(None, None, 2)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:-1, 0:2, ::2] = self.value


class TestSetValueItemSliceStep4(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:2:2, :] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(0, None), slice(1, 2, 2), slice(None)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:, 1:2:2, :] = self.value


# 1.2.3 step < 0
class TestSetValueItemSliceNegativeStep(TestSetValueApi):
    def set_shape(self):
        self.shape = [5, 2]

    def set_value(self):
        self.value = np.array([3, 4])

    def _call_setitem(self, x):
        x[5:2:-1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(5, 2, -1), self.value)
        return x

    def _get_answer(self):
        self.data[5:2:-1] = self.value


class TestSetValueItemSliceNegativeStep2(TestSetValueApi):
    def set_shape(self):
        self.shape = [5]

    def set_value(self):
        self.value = np.array([3, 4])

    def _call_setitem(self, x):
        x[1::-1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(1, None, -1), self.value)
        return x

    def _get_answer(self):
        self.data[1::-1] = self.value


class TestSetValueItemSliceNegativeStep3(TestSetValueApi):
    def set_shape(self):
        self.shape = [3]

    def set_value(self):
        self.value = np.array([3, 4, 5])

    def _call_setitem(self, x):
        x[::-1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(None, None, -1), self.value)
        return x

    def _get_answer(self):
        self.data[::-1] = self.value


class TestSetValueItemSliceNegativeStep4(TestSetValueApi):
    def set_shape(self):
        self.shape = [3, 4, 5]

    def _call_setitem(self, x):
        x[2:0:-1, 0:2, ::-1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(2, 0, -1), slice(0, 2), slice(None, None, -1)), self.value
        )
        return x

    def _get_answer(self):
        self.data[2:0:-1, 0:2, ::-1] = self.value


# 1.3 item is Ellipsis


class TestSetValueItemEllipsis1(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, ..., 1:] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(0, None), ..., slice(1, None)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:, ..., 1:] = self.value


class TestSetValueItemEllipsis2(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, ...] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (slice(0, None), ...), self.value)
        return x

    def _get_answer(self):
        self.data[0:, ...] = self.value


class TestSetValueItemEllipsis3(TestSetValueApi):
    def _call_setitem(self, x):
        x[..., 1:] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (..., slice(1, None)), self.value)
        return x

    def _get_answer(self):
        self.data[..., 1:] = self.value


class TestSetValueItemEllipsis4(TestSetValueApi):
    def _call_setitem(self, x):
        x[...] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, ..., self.value)
        return x

    def _get_answer(self):
        self.data[...] = self.value


# 1.4 item is Paddle Tensor
class TestSetValueItemTensor(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        x[zero] = self.value

    def _call_setitem_static_api(self, x):
        zero = paddle.full([], 0, dtype="int32")
        x = paddle.static.setitem(x, zero, self.value)
        return x

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueItemTensor2(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x[zero:two] = self.value

    def _call_setitem_static_api(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x = paddle.static.setitem(x, slice(zero, two), self.value)
        return x

    def _get_answer(self):
        self.data[0:2] = self.value


class TestSetValueItemTensor3(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x[zero:-1, 0:two] = self.value

    def _call_setitem_static_api(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x = paddle.static.setitem(
            x, (slice(zero, -1), slice(0, two)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:-1, 0:2] = self.value


class TestSetValueItemTensor4(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x[0:-1, zero:2, 0:6:two] = self.value

    def _call_setitem_static_api(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x = paddle.static.setitem(
            x, (slice(0, -1), slice(zero, 2), slice(0, 6, two)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:-1, 0:2, ::2] = self.value


class TestSetValueItemTensor5(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x[zero:, 1:2:two, :] = self.value

    def _call_setitem_static_api(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x = paddle.static.setitem(
            x, (slice(zero, None), slice(1, 2, two)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:, 1:2:2, :] = self.value


class TestSetValueItemTensor6(TestSetValueApi):
    def set_shape(self):
        self.shape = [3, 4, 5]

    def _call_setitem(self, x):
        minus1 = paddle.full([], -1, dtype="int32")
        zero = paddle.full([], 0, dtype="int32")
        x[2:zero:minus1, 0:2, 10:-6:minus1] = self.value

    def _call_setitem_static_api(self, x):
        minus1 = paddle.full([], -1, dtype="int32")
        zero = paddle.full([], 0, dtype="int64")
        x = paddle.static.setitem(
            x,
            (slice(2, zero, minus1), slice(0, 2), slice(10, -6, minus1)),
            self.value,
        )
        return x

    def _get_answer(self):
        self.data[2:0:-1, 0:2, ::-1] = self.value


# 1.5 item is None
class TestSetValueItemNone1(TestSetValueApi):
    def _call_setitem(self, x):
        x[None] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, None, self.value)
        return x

    def _get_answer(self):
        self.data[None] = self.value


class TestSetValueItemNone2(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, None, 1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (0, None, 1), self.value)
        return x

    def _get_answer(self):
        self.data[0, None, 1] = self.value


class TestSetValueItemNone3(TestSetValueApi):
    def _call_setitem(self, x):
        x[:, None, None, 1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (slice(None), None, None, 1), self.value)
        return x

    def _get_answer(self):
        self.data[:, None, None, 1] = self.value


class TestSetValueItemNone4(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, 0, None, 1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (0, 0, None, 1), self.value)
        return x

    def _get_answer(self):
        self.data[0, 0, None, 1] = self.value


class TestSetValueItemNone5(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, None, 0, None, 1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (0, None, 0, None, 1), self.value)
        return x

    def _get_answer(self):
        self.data[0, None, 0, None, 1] = self.value


class TestSetValueItemNone6(TestSetValueApi):
    def _call_setitem(self, x):
        x[None, 0, 0, None, 0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (None, 0, 0, None, 0), self.value)
        return x

    def _get_answer(self):
        self.data[None, 0, 0, None, 0] = self.value


class TestSetValueItemNone7(TestSetValueApi):
    def _call_setitem(self, x):
        x[:, None, 1] = np.zeros(self.shape)[:, None, 0]

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(None), None, 1), np.zeros(self.shape)[:, None, 0]
        )
        return x

    def _get_answer(self):
        self.data[:, None, 1] = np.zeros(self.shape)[:, None, 0]


class TestSetValueItemNone8(TestSetValueApi):
    def _call_setitem(self, x):
        x[:, 1, None] = np.zeros(self.shape)[:, 0, None]

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(None), 1, None), np.zeros(self.shape)[:, 0, None]
        )
        return x

    def _get_answer(self):
        self.data[:, 1, None] = np.zeros(self.shape)[:, 0, None]


class TestSetValueItemNone9(TestSetValueApi):
    def _call_setitem(self, x):
        x[None, :, 1, ..., None] = np.zeros(self.shape)[0, 0, :, None]

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x,
            (None, slice(None), 1, ..., None),
            np.zeros(self.shape)[0, 0, :, None],
        )
        return x

    def _get_answer(self):
        self.data[None, :, 1, ..., None] = np.zeros(self.shape)[0, 0, :, None]


class TestSetValueItemNone10(TestSetValueApi):
    def _call_setitem(self, x):
        x[..., None, :, None] = np.zeros(self.shape)[..., None, :, None]

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x,
            (..., None, slice(None), None),
            np.zeros(self.shape)[..., None, :, None],
        )
        return x

    def _get_answer(self):
        self.data[..., None, :, None] = np.zeros(self.shape)[..., None, :, None]


# 1.5 item is list or Tensor of bool
# NOTE(zoooo0820): Currently, 1-D List is same to Tuple.
# The semantic of index will be modified later.
class TestSetValueItemBool1(TestSetValueApi):
    def _call_setitem(self, x):
        x[[True, False]] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, [True, False], self.value)
        return x

    def _get_answer(self):
        self.data[[True, False]] = self.value


class TestSetValueItemBool2(TestSetValueApi):
    def _call_setitem(self, x):
        x[[False, False]] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, [False, False], self.value)
        return x

    def _get_answer(self):
        self.data[[False, False]] = self.value


class TestSetValueItemBool3(TestSetValueApi):
    def _call_setitem(self, x):
        x[[False, True]] = np.zeros(self.shape[2])

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, [False, True], np.zeros(self.shape[2]))
        return x

    def _get_answer(self):
        self.data[[False, True]] = np.zeros(self.shape[2])


class TestSetValueItemBool4(TestSetValueApi):
    def _call_setitem(self, x):
        idx = paddle.assign(np.array([False, True]))
        x[idx] = np.zeros(self.shape[2])

    def _call_setitem_static_api(self, x):
        idx = paddle.assign(np.array([False, True]))
        x = paddle.static.setitem(x, idx, np.zeros(self.shape[2]))
        return x

    def _get_answer(self):
        self.data[np.array([False, True])] = np.zeros(self.shape[2])


class TestSetValueItemBool5(TestSetValueApi):
    def _call_setitem(self, x):
        idx = paddle.assign(
            np.array([[False, True, False], [True, True, False]])
        )
        x[idx] = self.value

    def _call_setitem_static_api(self, x):
        idx = paddle.assign(
            np.array([[False, True, False], [True, True, False]])
        )
        x = paddle.static.setitem(x, idx, self.value)
        return x

    def _get_answer(self):
        self.data[np.array([[False, True, False], [True, True, False]])] = (
            self.value
        )


class TestSetValueItemBool6(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, ...] = 0
        x[x > 0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (0, ...), 0)
        x = paddle.static.setitem(x, x > 0, self.value)
        return x

    def _get_answer(self):
        self.data[0, ...] = 0
        self.data[self.data > 0] = self.value


# 2. Test different type of value: int, float, numpy.ndarray, Tensor
# 2.1 value is int32, int64, float32, float64, bool


def create_test_value_int32(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 7

        def set_dtype(self):
            self.dtype = "int32"

    cls_name = "{}_{}".format(parent.__name__, "ValueInt32")
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

    cls_name = "{}_{}".format(parent.__name__, "ValueInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_int64(TestSetValueItemInt)
create_test_value_int64(TestSetValueItemSlice)
create_test_value_int64(TestSetValueItemSlice2)
create_test_value_int64(TestSetValueItemSlice3)
create_test_value_int64(TestSetValueItemSlice4)


def create_test_value_fp16(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 3.7

        def set_dtype(self):
            self.dtype = "float16"

    cls_name = "{}_{}".format(parent.__name__, "Valuefp16")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_fp16(TestSetValueItemInt)
create_test_value_fp16(TestSetValueItemSlice)
create_test_value_fp16(TestSetValueItemSlice2)
create_test_value_fp16(TestSetValueItemSlice3)
create_test_value_fp16(TestSetValueItemSlice4)


def create_test_value_fp32(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 3.3

        def set_dtype(self):
            self.dtype = "float32"

    cls_name = "{}_{}".format(parent.__name__, "ValueFp32")
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

    cls_name = "{}_{}".format(parent.__name__, "ValueFp64")
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

    cls_name = "{}_{}".format(parent.__name__, "ValueBool")
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

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyInt32")
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

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyInt64")
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

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyFp32")
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

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyFp64")
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

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyBool")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_bool(TestSetValueItemInt)
create_test_value_numpy_bool(TestSetValueItemSlice)
create_test_value_numpy_bool(TestSetValueItemSlice2)
create_test_value_numpy_bool(TestSetValueItemSlice3)
create_test_value_numpy_bool(TestSetValueItemSlice4)


def create_test_value_complex64(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 42.1 + 42.1j

        def set_dtype(self):
            self.dtype = "complex64"

    cls_name = "{}_{}".format(parent.__name__, "ValueComplex64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_complex64(TestSetValueItemInt)
create_test_value_complex64(TestSetValueItemSlice)
create_test_value_complex64(TestSetValueItemSlice2)
create_test_value_complex64(TestSetValueItemSlice3)
create_test_value_complex64(TestSetValueItemSlice4)


def create_test_value_complex128(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = complex(
                np.finfo(np.float64).max + 1j * np.finfo(np.float64).min
            )

        def set_dtype(self):
            self.dtype = "complex128"

    cls_name = "{}_{}".format(parent.__name__, "ValueComplex128")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_complex128(TestSetValueItemInt)
create_test_value_complex128(TestSetValueItemSlice)
create_test_value_complex128(TestSetValueItemSlice2)
create_test_value_complex128(TestSetValueItemSlice3)
create_test_value_complex128(TestSetValueItemSlice4)


def create_test_value_numpy_complex64(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = np.array(42.1 + 42.1j)

        def set_dtype(self):
            self.dtype = "complex64"

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyComplex64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_complex64(TestSetValueItemInt)
create_test_value_numpy_complex64(TestSetValueItemSlice)
create_test_value_numpy_complex64(TestSetValueItemSlice2)
create_test_value_numpy_complex64(TestSetValueItemSlice3)
create_test_value_numpy_complex64(TestSetValueItemSlice4)


def create_test_value_numpy_complex128(parent):
    class TestValueInt(parent):
        def set_value(self):
            v = complex(
                np.finfo(np.float64).max + 1j * np.finfo(np.float64).min
            )
            self.value = np.array([v])

        def set_dtype(self):
            self.dtype = "complex128"

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyComplex128")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_complex128(TestSetValueItemInt)
create_test_value_numpy_complex128(TestSetValueItemSlice)
create_test_value_numpy_complex128(TestSetValueItemSlice2)
create_test_value_numpy_complex128(TestSetValueItemSlice3)
create_test_value_numpy_complex128(TestSetValueItemSlice4)


# 2.3 value is a Paddle Tensor (int32, int64, float32, float64, bool)
def create_test_value_tensor_int32(parent):
    class TestValueInt(parent):
        def set_dtype(self):
            self.dtype = "int32"

        def _call_setitem(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorInt32")
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
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorInt64")
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
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorFp32")
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
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorFp64")
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
            value = paddle.full(shape=[], fill_value=False, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=False, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = False

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorBool")
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

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, 0, self.value)
        return x

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueValueShape2(TestSetValueApi):
    def set_value(self):
        self.value = np.array([[3, 4, 5, 6]])  # shape is (1,4)

    def _call_setitem(self, x):
        x[0:1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(0, 1), self.value)
        return x

    def _get_answer(self):
        self.data[0:1] = self.value


class TestSetValueValueShape3(TestSetValueApi):
    def set_value(self):
        self.value = np.array(
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
        )  # shape is (3,4)

    def _call_setitem(self, x):
        x[0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, 0, self.value)
        return x

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueValueShape4(TestSetValueApi):
    def set_value(self):
        self.value = np.array(
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
        ).astype(
            self.dtype
        )  # shape is (3,4)

    def _call_setitem(self, x):
        x[0] = paddle.assign(self.value)  # x is Paddle.Tensor

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, 0, paddle.assign(self.value))
        return x

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueValueShape5(TestSetValueApi):
    def set_value(self):
        self.value = np.array([3, 3, 3]).astype(self.dtype)

    def set_shape(self):
        self.shape = [3, 4]

    def _call_setitem(self, x):
        x[:, 0] = paddle.assign(self.value)  # x is Paddle.Tensor

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(None), 0), paddle.assign(self.value)
        )
        return x

    def _get_answer(self):
        self.data[:, 0] = self.value


# This is to test case which dims of indexed Tensor is
# less than value Tensor on CPU / GPU.
class TestSetValueValueShape6(TestSetValueApi):
    def set_value(self):
        self.value = np.ones((1, 4)) * 5

    def set_shape(self):
        self.shape = [4, 4]

    def _call_setitem(self, x):
        x[:, 0] = self.value  # x is Paddle.Tensor

    def _get_answer(self):
        self.data[:, 0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (slice(None), 0), self.value)
        return x

    def test_api(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)

            static_out = self._run_static()
            dynamic_out = self._run_dynamic()
            self._get_answer()

            error_msg = (
                "\nIn {} mode: \nExpected res = \n{}, \n\nbut received : \n{}"
            )
            self.assertTrue(
                (self.data == static_out).all(),
                msg=error_msg.format("static", self.data, static_out),
            )
            self.assertTrue(
                (self.data == dynamic_out).all(),
                msg=error_msg.format("dynamic", self.data, dynamic_out),
            )


# 4. Test error
class TestError(TestSetValueBase):
    def _value_type_error(self):
        with self.assertRaisesRegex(
            TypeError,
            "Only support to assign an integer, float, numpy.ndarray or paddle.Tensor",
        ):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            value = [1]
            if paddle.in_dynamic_mode():
                x[0] = value
            else:
                x = paddle.static.setitem(x, 0, value)

    def _dtype_error(self):
        with self.assertRaisesRegex(
            TypeError,
            "When assign a numpy.ndarray, integer or float to a paddle.Tensor, ",
        ):
            y = paddle.ones(shape=self.shape, dtype="float16")
            y[0] = 1

    def _step_error(self):
        with self.assertRaisesRegex(ValueError, "step can not be 0"):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            if paddle.in_dynamic_mode():
                x[0:1:0] = self.value
            else:
                x = paddle.static.setitem(x, slice(0, 1, 0), self.value)

    def _ellipsis_error(self):
        with self.assertRaisesRegex(
            IndexError, "An index can only have a single ellipsis"
        ):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            x[..., ...] = self.value
        with self.assertRaisesRegex(ValueError, "the start or end is None"):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            one = paddle.ones([1])
            x[::one] = self.value

    def _bool_list_error(self):
        with self.assertRaises(IndexError):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            if paddle.in_dynamic_mode():
                x[[True, False], [True, False]] = 0
            else:
                x = paddle.static.setitem(x, ([True, False], [True, False]), 0)

    def _bool_tensor_error(self):
        with self.assertRaises(IndexError):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            idx = paddle.assign([True, False, True])
            if paddle.in_dynamic_mode():
                x[idx] = 0
            else:
                x = paddle.static.setitem(x, idx, 0)

    def _broadcast_mismatch(self):
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            value = np.array([3, 4, 5, 6, 7])
            x = paddle.static.setitem(x, 0, value)
        exe = paddle.static.Executor(paddle.CPUPlace())
        with self.assertRaises(ValueError):
            exe.run(program)

    def test_error(self):
        paddle.enable_static()
        with paddle.static.program_guard(self.program):
            self._value_type_error()
            self._bool_list_error()
            self._bool_tensor_error()
        self._broadcast_mismatch()


# 5. Test backward


class Model(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = paddle.nn.Conv2D(12, 12, 3)

    def forward(self, x, y):
        x = self.conv(x)
        y = self.conv(y)
        var = y.flatten()

        if paddle.in_dynamic_mode():
            x[0, :, 0, 0] = var
        else:
            x = paddle.static.setitem(x, (0, slice(None), 0, 0), var)
        loss = paddle.mean(x)
        return loss, var, x


class TestBackward(unittest.TestCase):
    def func_test_dynamic(self):
        model = Model()
        x = paddle.ones([1, 12, 3, 3]).astype("float32")
        y = paddle.ones([1, 12, 3, 3]).astype("float32")
        loss, var, x = model(x, y)
        loss.backward()

        self.assertTrue(var.grad.shape == x.grad[0, :, 0, 0].shape)
        self.assertTrue((0 == x.grad[0, :, 0, 0]).all())


class TestGradientTruncated(unittest.TestCase):
    def test_consistent_with_competitor(self):
        paddle.disable_static()

        def set_value(t, value):
            a = t * t
            a[0, 1] = value
            y = a * a
            return y.sum()

        # case 1
        array = np.arange(1, 1 + 2 * 3 * 4, dtype="float32").reshape(
            [1, 2, 1, 3, 1, 4]
        )
        value = np.arange(100, 104, dtype="float32").reshape(1, 4)

        inps = paddle.to_tensor(array, stop_gradient=False)
        value = paddle.to_tensor(value, stop_gradient=False)

        loss = set_value(inps, value)
        loss.backward()

        value_grad = np.array([[600.0, 606.0, 612.0, 618.0]])
        input_grad = np.array(
            [
                [
                    [
                        [
                            [[4.0, 32.0, 108.0, 256.0]],
                            [[500.0, 864.0, 1372.0, 2048.0]],
                            [[2916.0, 4000.0, 5324.0, 6912.0]],
                        ]
                    ],
                    [
                        [
                            [[0.0, 0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0, 0.0]],
                        ]
                    ],
                ]
            ]
        )
        np.testing.assert_array_equal(
            inps.grad.numpy(),
            input_grad,
            err_msg=f'The gradient of value should be \n{input_grad},\n but received {inps.grad.numpy()}',
        )
        np.testing.assert_array_equal(
            value.grad.numpy(),
            value_grad,
            err_msg=f'The gradient of input should be \n{value_grad},\n but received {value.grad.numpy()}',
        )

        # case 2
        array = np.arange(1, 2 * 3 * 4 + 1, dtype="float32").reshape([4, 2, 3])
        value = np.arange(100, 100 + 1, dtype="float32")

        inps2 = paddle.to_tensor(array, stop_gradient=False)
        value2 = paddle.to_tensor(value, stop_gradient=False)

        loss = set_value(inps2, value2)
        loss.backward()

        value_grad2 = np.array([600.0])
        input_grad2 = np.array(
            [
                [[4.0, 32.0, 108.0], [0.0, 0.0, 0.0]],
                [[1372.0, 2048.0, 2916.0], [4000.0, 5324.0, 6912.0]],
                [[8788.0, 10976.0, 13500.0], [16384.0, 19652.0, 23328.0]],
                [[27436.0, 32000.0, 37044.0], [42592.0, 48668.0, 55296.0]],
            ]
        )
        np.testing.assert_array_equal(
            inps2.grad.numpy(),
            input_grad2,
            err_msg=f'The gradient of value should be \n{input_grad},\n but received {inps2.grad.numpy()}',
        )
        np.testing.assert_array_equal(
            value2.grad.numpy(),
            value_grad2,
            err_msg=f'The gradient of input should be \n{value_grad},\n but received {value2.grad.numpy()}',
        )

        # case 3
        def set_value3(t, value):
            a = t * t
            a[0, :, 0, :] = value
            y = a * a
            return y.sum()

        array = np.arange(1, 1 + 2 * 3 * 4, dtype="float32").reshape(
            [4, 3, 1, 1, 2, 1]
        )
        value = np.arange(100, 100 + 2, dtype="float32").reshape(1, 2, 1)

        inps = paddle.to_tensor(array, stop_gradient=False)
        value = paddle.to_tensor(value, stop_gradient=False)

        loss = set_value3(inps, value)
        loss.backward()

        value_grad = np.array([[[600.0], [606.0]]])
        input_grad = np.array(
            [
                [[[[[0.0], [0.0]]]], [[[[0.0], [0.0]]]], [[[[0.0], [0.0]]]]],
                [
                    [[[[1372.0], [2048.0]]]],
                    [[[[2916.0], [4000.0]]]],
                    [[[[5324.0], [6912.0]]]],
                ],
                [
                    [[[[8788.0], [10976.0]]]],
                    [[[[13500.0], [16384.0]]]],
                    [[[[19652.0], [23328.0]]]],
                ],
                [
                    [[[[27436.0], [32000.0]]]],
                    [[[[37044.0], [42592.0]]]],
                    [[[[48668.0], [55296.0]]]],
                ],
            ]
        )
        np.testing.assert_array_equal(
            inps.grad.numpy(),
            input_grad,
            err_msg=f'The gradient of value should be \n{input_grad},\n but received {inps.grad.numpy()}',
        )
        np.testing.assert_array_equal(
            value.grad.numpy(),
            value_grad,
            err_msg=f'The gradient of input should be \n{value_grad},\n but received {value.grad.numpy()}',
        )

        # case 4: step >0
        def set_value4(t, value):
            a = t * t
            a[0, :, 0, ::3] = value
            y = a * a
            return y.sum()

        array = np.arange(1, 1 + 2 * 3 * 4, dtype="float32").reshape(
            [2, 3, 1, 4, 1]
        )
        value = np.arange(100, 100 + 2, dtype="float32").reshape(1, 2, 1)

        inps = paddle.to_tensor(array, stop_gradient=False)
        value = paddle.to_tensor(value, stop_gradient=False)

        loss = set_value4(inps, value)
        loss.backward()

        value_grad = np.array([[[600.0], [606.0]]])
        input_grad = np.array(
            [
                [
                    [[[0.0], [32.0], [108.0], [0.0]]],
                    [[[0.0], [864.0], [1372.0], [0.0]]],
                    [[[0.0], [4000.0], [5324.0], [0.0]]],
                ],
                [
                    [[[8788.0], [10976.0], [13500.0], [16384.0]]],
                    [[[19652.0], [23328.0], [27436.0], [32000.0]]],
                    [[[37044.0], [42592.0], [48668.0], [55296.0]]],
                ],
            ]
        )
        np.testing.assert_array_equal(
            inps.grad.numpy(),
            input_grad,
            err_msg=f'The gradient of value should be \n{input_grad},\n but received {inps.grad.numpy()}',
        )
        np.testing.assert_array_equal(
            value.grad.numpy(),
            value_grad,
            err_msg=f'The gradient of input should be \n{value_grad},\n but received {value.grad.numpy()}',
        )

        # case 5:a[0].shape==value.shape
        def set_value5(t, value):
            a = t * t
            a[0] = value
            y = a * a
            return y.sum()

        array = np.arange(1, 1 + 2 * 3 * 4, dtype="float32").reshape([2, 3, 4])
        value = np.arange(100, 100 + 12, dtype="float32").reshape(3, 4)

        inps = paddle.to_tensor(array, stop_gradient=False)
        value = paddle.to_tensor(value, stop_gradient=False)

        loss = set_value5(inps, value)
        loss.backward()

        value_grad = np.array(
            [
                [200.0, 202.0, 204.0, 206.0],
                [208.0, 210.0, 212.0, 214.0],
                [216.0, 218.0, 220.0, 222.0],
            ]
        )
        input_grad = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [8788.0, 10976.0, 13500.0, 16384.0],
                    [19652.0, 23328.0, 27436.0, 32000.0],
                    [37044.0, 42592.0, 48668.0, 55296.0],
                ],
            ]
        )
        np.testing.assert_array_equal(
            inps.grad.numpy(),
            input_grad,
            err_msg=f'The gradient of value should be \n{input_grad},\n but received {inps.grad.numpy()}',
        )
        np.testing.assert_array_equal(
            value.grad.numpy(),
            value_grad,
            err_msg=f'The gradient of input should be \n{value_grad},\n but received {value.grad.numpy()}',
        )

        # case 6: pass stop_gradient from value to x
        x = paddle.zeros([8, 8], dtype='float32')
        value = paddle.to_tensor([10], dtype='float32', stop_gradient=False)

        self.assertTrue(x.stop_gradient)
        self.assertTrue(x.is_leaf)

        x[0, :] = value

        self.assertTrue(not x.stop_gradient)
        self.assertTrue(not x.is_leaf)


class TestSetValueInplace(unittest.TestCase):
    def test_inplace(self):
        paddle.disable_static()
        with paddle.base.dygraph.guard():
            paddle.seed(100)
            a = paddle.rand(shape=[1, 4])
            a.stop_gradient = False
            b = a[:] * 1
            c = b
            b[paddle.zeros([], dtype='int32')] = 1.0

            self.assertTrue(id(b) == id(c))
            np.testing.assert_array_equal(b.numpy(), c.numpy())
            self.assertEqual(b.inplace_version, 1)

        paddle.enable_static()


class TestSetValueInplaceLeafVar(unittest.TestCase):
    def test_inplace_var_become_leaf_var(self):
        paddle.disable_static()

        a_grad_1, b_grad_1, a_grad_2, b_grad_2 = 0, 1, 2, 3
        with paddle.base.dygraph.guard():
            paddle.seed(100)
            a = paddle.rand(shape=[1, 4])
            b = paddle.rand(shape=[1, 4])
            a.stop_gradient = False
            b.stop_gradient = False
            c = a / b
            c.sum().backward()
            a_grad_1 = a.grad.numpy()
            b_grad_1 = b.grad.numpy()

        with paddle.base.dygraph.guard():
            paddle.seed(100)
            a = paddle.rand(shape=[1, 4])
            b = paddle.rand(shape=[1, 4])
            a.stop_gradient = False
            b.stop_gradient = False
            c = a / b
            d = paddle.zeros((4, 4))
            self.assertTrue(d.stop_gradient)
            d[0, :] = c
            self.assertFalse(d.stop_gradient)
            d[0, :].sum().backward()
            a_grad_2 = a.grad.numpy()
            b_grad_2 = b.grad.numpy()

        np.testing.assert_array_equal(a_grad_1, a_grad_2)
        np.testing.assert_array_equal(b_grad_1, b_grad_2)
        paddle.enable_static()


class TestSetValueIsSamePlace(unittest.TestCase):
    def test_is_same_place(self):
        paddle.disable_static()
        paddle.seed(100)
        paddle.set_device('cpu')
        a = paddle.rand(shape=[2, 3, 4])
        origin_place = a.place
        a[[0, 1], 1] = 10
        self.assertEqual(origin_place._type(), a.place._type())
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
        paddle.enable_static()


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestSetValueBFloat16(OpTest):
    def setUp(self):
        self.dtype = np.uint16
        self.shape = [22, 3, 4]
        self.op_type = 'set_value'
        self.data = np.ones(self.shape).astype(self.dtype)
        value = np.random.rand(4).astype('float32')

        expected_out = np.ones(self.shape).astype('float32')
        expected_out[0, 0] = value

        self.attrs = {
            'axes': [0, 1],
            'starts': [0, 0],
            'ends': [1, 1],
            'steps': [1, 1],
        }
        self.inputs = {
            'Input': convert_float_to_uint16(self.data),
            'ValueTensor': convert_float_to_uint16(value),
        }
        self.outputs = {'Out': convert_float_to_uint16(expected_out)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        # NOTE(zoooo0820) Here we set check_dygraph=False since set_value OP has no corresponding python api
        # to set self.python_api
        self.check_output_with_place(place, check_dygraph=False)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['Input'], 'Out', check_dygraph=False)


class TestSetValueWithScalarInStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.shape = (10, 2)
        self.exe = paddle.static.Executor()
        self.train_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()


class TestSetValueWithScalarInDygraph(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.shape = (10, 2)

    def test_value_input_is_scalar(self):
        x = paddle.ones(self.shape)
        x.stop_gradient = False
        y = x * 1

        # mock test case x[0, 0] = 10 with no ValueTensor input
        out = paddle._C_ops.set_value(
            y, [0, 0], [1, 1], [1, 1], [0, 1], [], [], [1], [10.0]
        )

        loss = out.sum()
        loss.backward()

        np_data = np.ones(self.shape).astype('float32')
        np_data[0, 0] = 10

        expected_x_grad = np.ones(self.shape)
        expected_x_grad[0, 0] = 0

        np.testing.assert_array_equal(out, np_data)
        np.testing.assert_array_equal(x.grad, expected_x_grad)


if __name__ == '__main__':
    unittest.main()
