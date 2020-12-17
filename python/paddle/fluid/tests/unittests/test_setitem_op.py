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

# Test setitem op in static mode

from __future__ import print_function

import unittest
import numpy as np

import paddle


class TestSetitemBase(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_value()
        self.set_dtype()
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


class TestSetitemApi(TestSetitemBase):
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
class TestSetitemItemInt(TestSetitemApi):
    def _call_setitem(self, x):
        x[0] = self.value

    def _get_answer(self):
        self.data[0] = self.value


class TestSetitemItemSlice(TestSetitemApi):
    def _call_setitem(self, x):
        x[0:2] = self.value

    def _get_answer(self):
        self.data[0:2] = self.value


class TestSetitemItemSlice2(TestSetitemApi):
    def _call_setitem(self, x):
        x[0:-1] = self.value

    def _get_answer(self):
        self.data[0:-1] = self.value


class TestSetitemItemSlice3(TestSetitemApi):
    def _call_setitem(self, x):
        x[0:-1, 0:2] = self.value

    def _get_answer(self):
        self.data[0:-1, 0:2] = self.value


class TestSetitemItemSlice4(TestSetitemApi):
    def _call_setitem(self, x):
        x[0:, 1:2, :] = self.value

    def _get_answer(self):
        self.data[0:, 1:2, :] = self.value


# 2. Test different type of value: int, float, numpy.ndarray, Tensor


def create_test_value_numpy(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = np.array([1])

    cls_name = "{0}_{1}".format(parent.__name__, "ValueInt")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy(TestSetitemItemInt)
create_test_value_numpy(TestSetitemItemSlice)
create_test_value_numpy(TestSetitemItemSlice2)
create_test_value_numpy(TestSetitemItemSlice3)
create_test_value_numpy(TestSetitemItemSlice4)


def create_test_value_tensor(parent):
    class TestValueInt(parent):
        def _call_setitem(self, x):
            value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{0}_{1}".format(parent.__name__, "ValueInt")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor(TestSetitemItemInt)
create_test_value_tensor(TestSetitemItemSlice)
create_test_value_tensor(TestSetitemItemSlice2)
create_test_value_tensor(TestSetitemItemSlice3)
create_test_value_tensor(TestSetitemItemSlice4)

# 3. Test different dtype: int32, int64, float32, bool


class TestSetitemDtypeInt32(TestSetitemApi):
    def set_dtype(self):
        self.dtype = "int32"


class TestSetitemDtypeInt64(TestSetitemApi):
    def set_dtype(self):
        self.dtype = "int64"


class TestSetitemDtypeFloat32(TestSetitemApi):
    def set_dtype(self):
        self.dtype = "float32"


class TestSetitemDtypeFloat64(TestSetitemApi):
    def set_dtype(self):
        self.dtype = "float64"

    def _call_setitem(self, x):
        value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
        x[0, 1] = value

    def _get_answer(self):
        self.data[0, 1] = 3


class TestSetitemDtypeBool(TestSetitemApi):
    def set_dtype(self):
        self.dtype = "bool"


class TestError(TestSetitemBase):
    def test_tensor(self):
        with paddle.static.program_guard(self.program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)

            with self.assertRaisesRegexp(
                    TypeError,
                    "Only support to assign an integer, float, numpy.ndarray or "
            ):
                value = [1]
                x[0] = value

            with self.assertRaisesRegexp(
                    TypeError,
                    "When assign a numpy.ndarray, integer or float to a paddle.Tensor, "
            ):
                y = paddle.ones(shape=self.shape, dtype="float64")
                y[0] = 1

            with self.assertRaisesRegexp(ValueError, "only support step is 1"):
                value = [1]
                x[0:1:2] = value


if __name__ == '__main__':
    unittest.main()
