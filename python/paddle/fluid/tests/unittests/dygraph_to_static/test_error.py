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

import os
import inspect
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.core import EnforceNotMet
from paddle.fluid.dygraph.dygraph_to_static import error
from paddle.fluid.dygraph.dygraph_to_static.origin_info import unwrap


def inner_func():
    fluid.layers.fill_constant(shape=[1, 2], value=9, dtype="int")
    return


@paddle.jit.to_static
def func_error_in_compile_time(x):
    x = fluid.dygraph.to_variable(x)
    inner_func()
    if fluid.layers.mean(x) < 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


@paddle.jit.to_static
def func_error_in_compile_time_2(x):
    x = fluid.dygraph.to_variable(x)
    x = fluid.layers.reshape(x, shape=[1, 2])
    return x


@paddle.jit.to_static
def func_error_in_runtime(x, iter_num=3):
    x = fluid.dygraph.to_variable(x)
    two = fluid.layers.fill_constant(shape=[1], value=2, dtype="int32")
    x = fluid.layers.reshape(x, shape=[1, two])
    return x


class TestErrorInCompileTime(unittest.TestCase):
    def setUp(self):
        self.set_func()
        self.set_input()
        self.set_exception_type()
        self.prog_trans = paddle.jit.ProgramTranslator()
        self.simplify_error = 1
        self.disable_error = 0

    def set_func(self):
        self.func = func_error_in_compile_time

    def set_exception_type(self):
        self.exception_type = TypeError

    def set_input(self):
        self.input = np.ones([3, 2])

    def set_message(self):
        self.expected_message = \
            ['File "{}", line 36, in func_error_in_compile_time'.format(self.filepath),
            'inner_func()',
            'File "{}", line 29, in inner_func'.format(self.filepath),
            'fluid.layers.fill_constant(shape=[1, 2], value=9, dtype="int")',
            ]

    def _test_create_message(self, error_data):
        self.filepath = inspect.getfile(unwrap(self.func))
        self.set_message()
        error_message = error_data.create_message()

        self.assertIn('In user code:', error_message)
        for m in self.expected_message:
            self.assertIn(m, error_message)

    def _test_attach_and_raise_new_exception(self, func_call):
        paddle.disable_static()
        with self.assertRaises(self.exception_type) as cm:
            func_call()
        exception = cm.exception

        error_data = getattr(exception, error.ERROR_DATA, None)

        self.assertIsInstance(error_data, error.ErrorData)
        self._test_create_message(error_data)

    def test_static_layer_call(self):
        # NOTE: self.func(self.input) is the StaticLayer().__call__(self.input)
        call_dy2static = lambda: self.func(self.input)

        self.set_flags(0)
        self._test_attach_and_raise_new_exception(call_dy2static)

    def test_program_translator_get_output(self):
        call_dy2static = lambda : self.prog_trans.get_output(unwrap(self.func), self.input)

        self.set_flags(0)
        self._test_attach_and_raise_new_exception(call_dy2static)

    def set_flags(self, disable_error=0, simplify_error=1):
        os.environ[error.DISABLE_ERROR_ENV_NAME] = str(disable_error)
        self.disable_error = int(os.getenv(error.DISABLE_ERROR_ENV_NAME, 0))
        self.assertEqual(self.disable_error, disable_error)

        os.environ[error.SIMPLIFY_ERROR_ENV_NAME] = str(simplify_error)
        self.simplify_error = int(os.getenv(error.SIMPLIFY_ERROR_ENV_NAME, 1))
        self.assertEqual(self.simplify_error, simplify_error)


class TestErrorInCompileTime2(TestErrorInCompileTime):
    def set_func(self):
        self.func = func_error_in_compile_time_2

    def set_exception_type(self):
        self.exception_type = EnforceNotMet

    def set_message(self):

        self.expected_message = \
            [
             'File "{}", line 47, in func_error_in_compile_time_2'.format(self.filepath),
             'x = fluid.layers.reshape(x, shape=[1, 2])'
             ]


class TestErrorInRuntime(TestErrorInCompileTime):
    def set_func(self):
        self.func = func_error_in_runtime

    def set_exception_type(self):
        self.exception_type = EnforceNotMet

    def set_message(self):
        self.expected_message = \
            [
                'File "{}", line 55, in func_error_in_runtime'.format(self.filepath),
                'x = fluid.layers.reshape(x, shape=[1, two])'
            ]

    def _test_create_message(self, error_data):
        self.filepath = inspect.getfile(unwrap(self.func))
        self.set_message()

        with self.assertRaises(ValueError):
            error_data.create_message()

        error_data.in_runtime = False
        error_message = error_data.create_message()

        self.assertIn('In user code:', error_message)
        for m in self.expected_message:
            self.assertIn(m, error_message)


@unwrap
@paddle.jit.to_static()
def func_decorated_by_other_1():
    return 1


@paddle.jit.to_static()
@unwrap
def func_decorated_by_other_2():
    return 1


class TestErrorInOther(unittest.TestCase):
    def test(self):
        paddle.disable_static()
        prog_trans = paddle.jit.ProgramTranslator()
        with self.assertRaises(NotImplementedError):
            prog_trans.get_output(func_decorated_by_other_1)

        with self.assertRaises(NotImplementedError):
            func_decorated_by_other_2()


if __name__ == '__main__':
    unittest.main()
