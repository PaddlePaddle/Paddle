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

from __future__ import print_function

import unittest
import numpy as np

import paddle


class TestSetitemBase(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_value()
        self.set_dtype()
        self.shape = [2, 2, 3]
        self.data = np.ones(self.shape).astype(self.dtype)
        self.program = paddle.static.Program()

    def set_value(self):
        self.value = 0

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


class TestSetitem2(TestSetitemApi):
    def set_value(self):
        self.value = np.array([1])


class TestSetitem3(TestSetitemApi):
    def set_value(self):
        self.value = np.array([1])


class TestSetitem4(TestSetitemApi):
    def _call_setitem(self, x):
        x[0:2] = self.value

    def _get_answer(self):
        self.data[0:2] = self.value


class TestSetitem5(TestSetitemApi):
    def _call_setitem(self, x):
        x[0:2, 1] = self.value

    def _get_answer(self):
        self.data[0:2, 1] = self.value


class TestSetitem6(TestSetitemApi):
    def set_dtype(self):
        self.dtype = "int32"


class TestSetitem7(TestSetitemApi):
    def set_dtype(self):
        self.dtype = "int64"


# TODO(liym27): support bool
# class TestSetitem8(TestSetitemApi):
#     def set_dtype(self):
#         self.dtype = "bool"


class TestSetitem9(TestSetitemApi):
    def _call_setitem(self, x):
        value = paddle.full(shape=[1], fill_value=3, dtype=self.dtype)
        x[0, 1] = value

    def _get_answer(self):
        self.data[0, 1] = 3


class TestError(TestSetitemBase):
    def test_tensor(self):
        with paddle.static.program_guard(self.program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)

            with self.assertRaises(TypeError):
                value = [1]
                x[0] = value

            with self.assertRaises(TypeError):
                y = paddle.ones(shape=self.shape, dtype="float64")
                y[0] = 1


if __name__ == '__main__':
    unittest.main()
