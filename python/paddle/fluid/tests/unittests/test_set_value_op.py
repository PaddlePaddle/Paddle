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
from paddle.fluid.layer_helper import LayerHelper
from functools import reduce
from paddle.fluid.framework import _test_eager_guard, _in_legacy_dygraph


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

    def _get_answer(self):
        self.data[0, 0] = self.value


class TestSetValueApi(TestSetValueBase):
    def _run_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(self.program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            self._call_setitem(x)

        exe = paddle.static.Executor(paddle.CPUPlace())
        out = exe.run(self.program, fetch_list=[x])
        paddle.disable_static()
        return out

    def _run_dynamic(self):
        paddle.disable_static()
        x = paddle.ones(shape=self.shape, dtype=self.dtype)
        self._call_setitem(x)
        out = x.numpy()
        paddle.enable_static()
        return out

    def func_test_api(self):
        static_out = self._run_static()
        dynamic_out = self._run_dynamic()
        self._get_answer()

        error_msg = "\nIn {} mode: \nExpected res = \n{}, \n\nbut received : \n{}"
        self.assertTrue(
            (self.data == static_out).all(),
            msg=error_msg.format("static", self.data, static_out))
        self.assertTrue(
            (self.data == dynamic_out).all(),
            msg=error_msg.format("dynamic", self.data, dynamic_out))

    def test_api(self):
        with _test_eager_guard():
            self.func_test_api()
        self.func_test_api()


# 1. Test different type of item: int, Python slice, Paddle Tensor
# 1.1 item is int
class TestSetValueItemInt(TestSetValueApi):
    def _call_setitem(self, x):
        x[0] = self.value

    def _get_answer(self):
        self.data[0] = self.value


# 1.2 item is slice
# 1.2.1 step is 1
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


class TestSetValueItemSlice5(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:1, :] = self.value

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

        i = paddle.zeros(shape=(1, ), dtype='int32')
        i, x = paddle.fluid.layers.while_loop(cond, body, [i, x])

    def _get_answer(self):
        self.data[0] = self.value


# 1.2.2 step > 1
class TestSetValueItemSliceStep(TestSetValueApi):
    def set_shape(self):
        self.shape = [5, 5, 5]

    def _call_setitem(self, x):
        x[0:2:2] = self.value

    def _get_answer(self):
        self.data[0:2:2] = self.value


class TestSetValueItemSliceStep2(TestSetValueApi):
    def set_shape(self):
        self.shape = [7, 5, 5]

    def _call_setitem(self, x):
        x[0:-1:3] = self.value

    def _get_answer(self):
        self.data[0:-1:3] = self.value


class TestSetValueItemSliceStep3(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1, 0:2, ::2] = self.value

    def _get_answer(self):
        self.data[0:-1, 0:2, ::2] = self.value


class TestSetValueItemSliceStep4(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:2:2, :] = self.value

    def _get_answer(self):
        self.data[0:, 1:2:2, :] = self.value


# 1.2.3 step < 0
class TestSetValueItemSliceNegetiveStep(TestSetValueApi):
    def set_shape(self):
        self.shape = [5, 2]

    def set_value(self):
        self.value = np.array([3, 4])

    def _call_setitem(self, x):
        x[5:2:-1] = self.value

    def _get_answer(self):
        self.data[5:2:-1] = self.value


class TestSetValueItemSliceNegetiveStep2(TestSetValueApi):
    def set_shape(self):
        self.shape = [5]

    def set_value(self):
        self.value = np.array([3, 4])

    def _call_setitem(self, x):
        x[1::-1] = self.value

    def _get_answer(self):
        self.data[1::-1] = self.value


class TestSetValueItemSliceNegetiveStep3(TestSetValueApi):
    def set_shape(self):
        self.shape = [3]

    def set_value(self):
        self.value = np.array([3, 4, 5])

    def _call_setitem(self, x):
        x[::-1] = self.value

    def _get_answer(self):
        self.data[::-1] = self.value


class TestSetValueItemSliceNegetiveStep4(TestSetValueApi):
    def set_shape(self):
        self.shape = [3, 4, 5]

    def _call_setitem(self, x):
        x[2:0:-1, 0:2, ::-1] = self.value

    def _get_answer(self):
        self.data[2:0:-1, 0:2, ::-1] = self.value


# 1.3 item is Ellipsis


class TestSetValueItemEllipsis1(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, ..., 1:] = self.value

    def _get_answer(self):
        self.data[0:, ..., 1:] = self.value


class TestSetValueItemEllipsis2(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, ...] = self.value

    def _get_answer(self):
        self.data[0:, ...] = self.value


class TestSetValueItemEllipsis3(TestSetValueApi):
    def _call_setitem(self, x):
        x[..., 1:] = self.value

    def _get_answer(self):
        self.data[..., 1:] = self.value


class TestSetValueItemEllipsis4(TestSetValueApi):
    def _call_setitem(self, x):
        x[...] = self.value

    def _get_answer(self):
        self.data[...] = self.value


# 1.4 item is Paddle Tensor
class TestSetValueItemTensor(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([1], 0, dtype="int32")
        x[zero] = self.value

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueItemTensor2(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([1], 0, dtype="int32")
        two = paddle.full([1], 2, dtype="int64")
        x[zero:two] = self.value

    def _get_answer(self):
        self.data[0:2] = self.value


class TestSetValueItemTensor3(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([1], 0, dtype="int32")
        two = paddle.full([1], 2, dtype="int64")
        x[zero:-1, 0:two] = self.value

    def _get_answer(self):
        self.data[0:-1, 0:2] = self.value


class TestSetValueItemTensor4(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([1], 0, dtype="int32")
        two = paddle.full([1], 2, dtype="int64")
        x[0:-1, zero:2, 0:6:two] = self.value

    def _get_answer(self):
        self.data[0:-1, 0:2, ::2] = self.value


class TestSetValueItemTensor5(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([1], 0, dtype="int32")
        two = paddle.full([1], 2, dtype="int64")
        x[zero:, 1:2:two, :] = self.value

    def _get_answer(self):
        self.data[0:, 1:2:2, :] = self.value


class TestSetValueItemTensor6(TestSetValueApi):
    def set_shape(self):
        self.shape = [3, 4, 5]

    def _call_setitem(self, x):
        minus1 = paddle.full([1], -1, dtype="int32")
        zero = paddle.full([1], 0, dtype="int32")
        x[2:zero:minus1, 0:2, 10:-6:minus1] = self.value

    def _get_answer(self):
        self.data[2:0:-1, 0:2, ::-1] = self.value


# 1.5 item is None
class TestSetValueItemNone1(TestSetValueApi):
    def _call_setitem(self, x):
        x[None] = self.value

    def _get_answer(self):
        self.data[None] = self.value


class TestSetValueItemNone2(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, None, 1] = self.value

    def _get_answer(self):
        self.data[0, None, 1] = self.value


class TestSetValueItemNone3(TestSetValueApi):
    def _call_setitem(self, x):
        x[:, None, None, 1] = self.value

    def _get_answer(self):
        self.data[:, None, None, 1] = self.value


class TestSetValueItemNone4(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, 0, None, 1] = self.value

    def _get_answer(self):
        self.data[0, 0, None, 1] = self.value


class TestSetValueItemNone5(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, None, 0, None, 1] = self.value

    def _get_answer(self):
        self.data[0, None, 0, None, 1] = self.value


class TestSetValueItemNone6(TestSetValueApi):
    def _call_setitem(self, x):
        x[None, 0, 0, None, 0] = self.value

    def _get_answer(self):
        self.data[None, 0, 0, None, 0] = self.value


class TestSetValueItemNone7(TestSetValueApi):
    def _call_setitem(self, x):
        x[:, None, 1] = np.zeros(self.shape)[:, None, 0]

    def _get_answer(self):
        self.data[:, None, 1] = np.zeros(self.shape)[:, None, 0]


class TestSetValueItemNone8(TestSetValueApi):
    def _call_setitem(self, x):
        x[:, 1, None] = np.zeros(self.shape)[:, 0, None]

    def _get_answer(self):
        self.data[:, 1, None] = np.zeros(self.shape)[:, 0, None]


class TestSetValueItemNone9(TestSetValueApi):
    def _call_setitem(self, x):
        x[None, :, 1, ..., None] = np.zeros(self.shape)[0, 0, :, None]

    def _get_answer(self):
        self.data[None, :, 1, ..., None] = np.zeros(self.shape)[0, 0, :, None]


class TestSetValueItemNone10(TestSetValueApi):
    def _call_setitem(self, x):
        x[..., None, :, None] = np.zeros(self.shape)[..., None, :, None]

    def _get_answer(self):
        self.data[..., None, :, None] = np.zeros(self.shape)[..., None, :, None]


# 1.5 item is list or Tensor of bol
class TestSetValueItemBool1(TestSetValueApi):
    def _call_setitem(self, x):
        x[[True, False]] = self.value

    def _get_answer(self):
        self.data[[True, False]] = self.value


class TestSetValueItemBool2(TestSetValueApi):
    def _call_setitem(self, x):
        x[[False, False]] = self.value

    def _get_answer(self):
        self.data[[False, False]] = self.value


class TestSetValueItemBool3(TestSetValueApi):
    def _call_setitem(self, x):
        x[[False, True]] = np.zeros(self.shape[2])

    def _get_answer(self):
        self.data[[False, True]] = np.zeros(self.shape[2])


class TestSetValueItemBool4(TestSetValueApi):
    def _call_setitem(self, x):
        idx = paddle.assign(np.array([False, True]))
        x[idx] = np.zeros(self.shape[2])

    def _get_answer(self):
        self.data[np.array([False, True])] = np.zeros(self.shape[2])


class TestSetValueItemBool5(TestSetValueApi):
    def _call_setitem(self, x):
        idx = paddle.assign(
            np.array([[False, True, False], [True, True, False]]))
        x[idx] = self.value

    def _get_answer(self):
        self.data[np.array([[False, True, False], [True, True, False]
                            ])] = self.value


class TestSetValueItemBool6(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, ...] = 0
        x[x > 0] = self.value

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


class TestSetValueValueShape5(TestSetValueApi):
    def set_value(self):
        self.value = np.array([3, 3, 3]).astype(self.dtype)

    def set_shape(self):
        self.shape = [3, 4]

    def _call_setitem(self, x):
        x[:, 0] = paddle.assign(self.value)  # x is Paddle.Tensor

    def _get_answer(self):
        self.data[:, 0] = self.value


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
        with self.assertRaisesRegexp(ValueError, "step can not be 0"):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            x[0:1:0] = self.value

    def _ellipsis_error(self):
        with self.assertRaisesRegexp(
                IndexError, "An index can only have a single ellipsis"):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            x[..., ...] = self.value
        with self.assertRaisesRegexp(ValueError, "the start or end is None"):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            one = paddle.ones([1])
            x[::one] = self.value

    def _bool_list_error(self):
        with self.assertRaises(TypeError):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            x[[True, False, 0]] = 0

        with self.assertRaises(IndexError):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            x[[True, False], [True, False]] = 0

    def _bool_tensor_error(self):
        with self.assertRaises(IndexError):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            idx = paddle.assign([True, False, True])
            x[idx] = 0

    def _broadcast_mismatch(self):
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            value = np.array([3, 4, 5, 6, 7])
            x[0] = value
        exe = paddle.static.Executor(paddle.CPUPlace())
        with self.assertRaises(ValueError):
            exe.run(program)

    def test_error(self):
        paddle.enable_static()
        with paddle.static.program_guard(self.program):
            self._value_type_error()
            self._dtype_error()
            self._step_error()
            self._bool_list_error()
            self._bool_tensor_error()
        self._broadcast_mismatch()


# 5. Test backward


class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = paddle.nn.Conv2D(12, 12, 3)

    def forward(self, x, y):
        x = self.conv(x)
        y = self.conv(y)
        var = y.flatten()

        x[0, :, 0, 0] = var
        loss = paddle.mean(x)
        return loss, var, x


class TestBackward(unittest.TestCase):
    def test_static(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        x_np = np.random.random(size=(4, 4)).astype('float32')
        y_np = np.random.random(size=(4, 4)).astype('float32')
        label_np = np.random.randint(2, size=(4, 1)).astype('int64')

        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="x", shape=[4, 4], dtype='float32')
            y = paddle.static.data(name="y", shape=[4, 4], dtype='float32')

            label = paddle.static.data(
                name="label", shape=[4, 1], dtype='int64')

            z = paddle.add(x, y)
            var = y[0, :]
            z[0, :] = var

            prediction = paddle.static.nn.fc(x=z, size=2, activation='softmax')

            cost = paddle.nn.functional.cross_entropy(
                input=prediction, label=label)
            loss = paddle.mean(cost)
            sgd = paddle.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(startup_program)

        var_grad, z_grad = exe.run(
            main_program,
            feed={"x": x_np,
                  "y": y_np,
                  "label": label_np},
            fetch_list=[var.name + "@GRAD", z.name + "@GRAD"])

        self.assertTrue((var_grad == z_grad[0, :]).all())
        paddle.disable_static()

    def func_test_dynamic(self):
        model = Model()
        x = paddle.ones([1, 12, 3, 3]).astype("float32")
        y = paddle.ones([1, 12, 3, 3]).astype("float32")
        loss, var, x = model(x, y)
        loss.backward()

        self.assertTrue(var.grad.shape == x.grad[0, :, 0, 0].shape)
        # 
        # TODO(pangyoki) add inplace and delete if
        if _in_legacy_dygraph():
            self.assertTrue((0 == x.grad[0, :, 0, 0]).all())

    def test_dynamic(self):
        with _test_eager_guard():
            self.func_test_dynamic()
        self.func_test_dynamic()


class TestGradientTruncated(unittest.TestCase):
    def func_test_consistent_with_competitor(self):
        paddle.disable_static()

        def set_value(t, value):
            a = t * t
            a[0, 1] = value
            y = a * a
            return y.sum()

        # case 1
        array = np.arange(
            1, 1 + 2 * 3 * 4, dtype="float32").reshape([1, 2, 1, 3, 1, 4])
        value = np.arange(100, 104, dtype="float32").reshape(1, 4)

        inps = paddle.to_tensor(array, stop_gradient=False)
        value = paddle.to_tensor(value, stop_gradient=False)

        loss = set_value(inps, value)
        loss.backward()

        value_grad = np.array([[600., 606., 612., 618.]])
        input_grad = np.array(
            [[[[[[4., 32., 108., 256.]], [[500., 864., 1372., 2048.]],
                [[2916., 4000., 5324., 6912.]]]],
              [[[[0., 0., 0., 0.]], [[0., 0., 0., 0.]], [[0., 0., 0., 0.]]]]]])
        self.assertTrue(
            np.array_equal(inps.grad.numpy(), input_grad),
            msg="The gradient of value should be \n{},\n but reveived {}".
            format(input_grad, inps.grad.numpy()))
        self.assertTrue(
            np.array_equal(value.grad.numpy(), value_grad),
            msg="The gradient of input should be \n{},\n but reveived {}".
            format(value_grad, value.grad.numpy()))

        # case 2
        array = np.arange(1, 2 * 3 * 4 + 1, dtype="float32").reshape([4, 2, 3])
        value = np.arange(100, 100 + 1, dtype="float32")

        inps2 = paddle.to_tensor(array, stop_gradient=False)
        value2 = paddle.to_tensor(value, stop_gradient=False)

        loss = set_value(inps2, value2)
        loss.backward()

        value_grad2 = np.array([600.])
        input_grad2 = np.array(
            [[[4., 32., 108.], [0., 0., 0.]], [[1372., 2048., 2916.],
                                               [4000., 5324., 6912.]],
             [[8788., 10976., 13500.], [16384., 19652., 23328.]],
             [[27436., 32000., 37044.], [42592., 48668., 55296.]]])
        self.assertTrue(
            np.array_equal(inps2.grad.numpy(), input_grad2),
            msg="The gradient of value should be \n{},\n but reveived {}".
            format(input_grad, inps2.grad.numpy()))
        self.assertTrue(
            np.array_equal(value2.grad.numpy(), value_grad2),
            msg="The gradient of input should be \n{},\n but reveived {}".
            format(value_grad, value2.grad.numpy()))

        # case 3
        def set_value3(t, value):
            a = t * t
            a[0, :, 0, :] = value
            y = a * a
            return y.sum()

        array = np.arange(
            1, 1 + 2 * 3 * 4, dtype="float32").reshape([4, 3, 1, 1, 2, 1])
        value = np.arange(100, 100 + 2, dtype="float32").reshape(1, 2, 1)

        inps = paddle.to_tensor(array, stop_gradient=False)
        value = paddle.to_tensor(value, stop_gradient=False)

        loss = set_value3(inps, value)
        loss.backward()

        value_grad = np.array([[[600.], [606.]]])
        input_grad = np.array(
            [[[[[[0.], [0.]]]], [[[[0.], [0.]]]], [[[[0.], [0.]]]]],
             [[[[[1372.], [2048.]]]], [[[[2916.], [4000.]]]],
              [[[[5324.], [6912.]]]]], [[[[[8788.], [10976.]]]], [[[[13500.],
                                                                    [16384.]]]],
                                        [[[[19652.], [23328.]]]]],
             [[[[[27436.], [32000.]]]], [[[[37044.], [42592.]]]],
              [[[[48668.], [55296.]]]]]])
        self.assertTrue(
            np.array_equal(inps.grad.numpy(), input_grad),
            msg="The gradient of value should be \n{},\n but reveived {}".
            format(input_grad, inps.grad.numpy()))
        self.assertTrue(
            np.array_equal(value.grad.numpy(), value_grad),
            msg="The gradient of input should be \n{},\n but reveived {}".
            format(value_grad, value.grad.numpy()))

        #case 4: step >0
        def set_value4(t, value):
            a = t * t
            a[0, :, 0, ::3] = value
            y = a * a
            return y.sum()

        array = np.arange(
            1, 1 + 2 * 3 * 4, dtype="float32").reshape([2, 3, 1, 4, 1])
        value = np.arange(100, 100 + 2, dtype="float32").reshape(1, 2, 1)

        inps = paddle.to_tensor(array, stop_gradient=False)
        value = paddle.to_tensor(value, stop_gradient=False)

        loss = set_value4(inps, value)
        loss.backward()

        value_grad = np.array([[[600.], [606.]]])
        input_grad = np.array([[[[[0.], [32.], [108.],
                                  [0.]]], [[[0.], [864.], [1372.], [0.]]],
                                [[[0.], [4000.], [5324.], [0.]]]],
                               [[[[8788.], [10976.], [13500.], [16384.]]],
                                [[[19652.], [23328.], [27436.], [32000.]]],
                                [[[37044.], [42592.], [48668.], [55296.]]]]])
        self.assertTrue(
            np.array_equal(inps.grad.numpy(), input_grad),
            msg="The gradient of value should be \n{},\n but reveived {}".
            format(input_grad, inps.grad.numpy()))
        self.assertTrue(
            np.array_equal(value.grad.numpy(), value_grad),
            msg="The gradient of input should be \n{},\n but reveived {}".
            format(value_grad, value.grad.numpy()))

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

        value_grad = np.array([[200., 202., 204., 206.],
                               [208., 210., 212., 214.],
                               [216., 218., 220., 222.]])
        input_grad = np.array([[[0., 0., 0., 0.], [0., 0., 0., 0.],
                                [0., 0., 0., 0.]],
                               [[8788., 10976., 13500., 16384.],
                                [19652., 23328., 27436., 32000.],
                                [37044., 42592., 48668., 55296.]]])
        self.assertTrue(
            np.array_equal(inps.grad.numpy(), input_grad),
            msg="The gradient of value should be \n{},\n but reveived {}".
            format(input_grad, inps.grad.numpy()))
        self.assertTrue(
            np.array_equal(value.grad.numpy(), value_grad),
            msg="The gradient of input should be \n{},\n but reveived {}".
            format(value_grad, value.grad.numpy()))

        # case 6: pass stop_gradient from value to x
        x = paddle.zeros([8, 8], dtype='float32')
        value = paddle.to_tensor([10], dtype='float32', stop_gradient=False)

        self.assertTrue(x.stop_gradient)
        self.assertTrue(x.is_leaf)

        x[0, :] = value

        self.assertTrue(~x.stop_gradient)
        self.assertTrue(~x.is_leaf)

    def test_consistent_with_competitor(self):
        with _test_eager_guard():
            self.func_test_consistent_with_competitor()
        self.func_test_consistent_with_competitor()

    def test_static_graph(self):
        paddle.enable_static()

        to_string = lambda x, i, : x + '_' + str(i)
        numel = lambda input_shape: reduce(lambda x, y: x * y, input_shape)

        def op1(x):
            value = paddle.fluid.layers.fill_constant([1], "float32", 1)
            # test stop_gradient 
            value.stop_gradient = True
            x.stop_gradient = False
            start = paddle.fluid.layers.fill_constant(
                [1], "int32", 5, force_cpu=True)
            end = paddle.fluid.layers.fill_constant(
                [1], "int32", 0, force_cpu=True)
            step = paddle.fluid.layers.fill_constant(
                [1], "int32", -2, force_cpu=True)

            inputs = {
                'Input': x,
                'ValueTensor': value,
                'StartsTensorList': [start, ],
                'EndsTensorList': [end, ],
                'StepsTensorList': [step, ]
            }

            helper = LayerHelper("set_value")
            y = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(
                type="set_value",
                inputs=inputs,
                outputs={'Out': y},
                attrs={'axes': [0]})

            return y, value

        def op2(x):
            value = paddle.fluid.layers.fill_constant([1, 3, 2], "float32", 1)
            # test stop_gradient 
            value.stop_gradient = False
            x.stop_gradient = False
            attrs = {
                'axes': [0],
                'starts': [6],
                'ends': [0],
                'steps': [-4],
                'decrease_axes': [],
                'none_axes': [],
                'dtype': paddle.float32
            }
            inputs = {'Input': x, 'ValueTensor': value}

            helper = LayerHelper("set_value")
            y = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(
                type="set_value",
                inputs=inputs,
                outputs={'Out': y},
                attrs=attrs)

            return y, value

        def op3(x):
            value = paddle.fluid.layers.fill_constant([1], "float32", 1)
            x.stop_gradient = True
            value.stop_gradient = False
            start = paddle.fluid.layers.fill_constant(
                [1], "int32", 0, force_cpu=True)
            end = paddle.fluid.layers.fill_constant(
                [1], "int32", 5, force_cpu=True)
            step = paddle.fluid.layers.fill_constant(
                [1], "int32", 3, force_cpu=True)

            inputs = {
                'Input': x,
                'ValueTensor': value,
                'StartsTensorList': [start, ],
                'EndsTensorList': [end, ],
                'StepsTensorList': [step, ]
            }

            helper = LayerHelper("set_value")
            y = helper.create_variable_for_type_inference(dtype=x.dtype)

            helper.append_op(
                type="set_value",
                inputs=inputs,
                outputs={'Out': y},
                attrs={'axes': [0]})

            return y, value

        def set_value(array, i, op):
            name_x = to_string('x', i)
            x = paddle.static.data(
                name=name_x, shape=array.shape, dtype='float32')

            # set_value_op in __get/setitem__ is an inplace operation. 
            # When `input.stop_gradient = True` and `value.stop_gradient = False`, 
            # set_value_grad_op will not be run during backward.
            y, value = op(x)

            y2 = y + 1
            loss = paddle.fluid.layers.reduce_sum(y2)
            sgd = paddle.optimizer.Adam()
            sgd.minimize(loss)
            place = paddle.fluid.CPUPlace(
            ) if not paddle.fluid.core.is_compiled_with_cuda(
            ) else paddle.fluid.CUDAPlace(0)

            prog = paddle.static.default_main_program()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            fetch_list = []
            if not x.stop_gradient:
                fetch_list.append(x.grad_name)
            if not value.stop_gradient:
                fetch_list.append(value.grad_name)
            out = exe.run(prog, feed={x.name: array}, fetch_list=fetch_list)
            return out

        input_shape = [7, 6, 5, 4, 3, 2]

        array = np.arange(
            0, numel(input_shape), dtype="float32").reshape(input_shape)

        for i in range(len(input_shape)):
            program = paddle.static.Program()
            with paddle.static.program_guard(program):
                out1 = set_value(array, i, op1)
                self.assertTrue((out1[0][5:0:-2] == 0).all())

            if len(array.shape) > 2:
                program2 = paddle.static.Program()
                with paddle.static.program_guard(program2):
                    out2 = set_value(array, i, op2)
                    self.assertTrue((out2[0][6:0:-4] == 0).all())

            program3 = paddle.static.Program()
            with paddle.static.program_guard(program3):
                out3 = set_value(array, i, op3)
                self.assertTrue((numel(out1[0][0:5:3].shape) == out3[0]).all())

            array = array[0]
        paddle.disable_static()


class TestSetValueInplace(unittest.TestCase):
    def test_inplace(self):
        paddle.disable_static()
        with paddle.fluid.dygraph.guard():
            paddle.seed(100)
            a = paddle.rand(shape=[1, 4])
            a.stop_gradient = False
            b = a[:]
            c = b
            b[paddle.to_tensor(0)] = 1.0

            self.assertTrue(id(b) == id(c))
            self.assertTrue(np.array_equal(b.numpy(), c.numpy()))
            self.assertEqual(b.inplace_version, 1)

        paddle.enable_static()


class TestSetValueInplaceLeafVar(unittest.TestCase):
    def test_inplace_var_become_leaf_var(self):
        paddle.disable_static()

        a_grad_1, b_grad_1, a_grad_2, b_grad_2 = 0, 1, 2, 3
        with paddle.fluid.dygraph.guard():
            paddle.seed(100)
            a = paddle.rand(shape=[1, 4])
            b = paddle.rand(shape=[1, 4])
            a.stop_gradient = False
            b.stop_gradient = False
            c = a / b
            c.sum().backward()
            a_grad_1 = a.grad.numpy()
            b_grad_1 = b.grad.numpy()

        with paddle.fluid.dygraph.guard():
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

        self.assertTrue(np.array_equal(a_grad_1, a_grad_2))
        self.assertTrue(np.array_equal(b_grad_1, b_grad_2))
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
