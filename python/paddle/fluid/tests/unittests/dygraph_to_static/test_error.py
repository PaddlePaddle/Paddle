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

import inspect
import unittest

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.core import EnforceNotMet
from paddle.fluid.dygraph.dygraph_to_static.error import ERROR_DATA, ErrorData
from paddle.fluid.dygraph.dygraph_to_static.origin_info import unwrap
from paddle.fluid.dygraph.jit import declarative


def inner_func():
    fluid.layers.fill_constant(shape=[1, 2], value=9, dtype="int")
    return


@declarative
def func_error_in_compile_time(x):
    x = fluid.dygraph.to_variable(x)
    inner_func()
    if fluid.layers.mean(x) < 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


@declarative
def func_error_in_compile_time_2(x):
    x = fluid.dygraph.to_variable(x)
    x = fluid.layers.reshape(x, shape=[1, 2])
    return x


@declarative
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

    def test(self):
        with fluid.dygraph.guard():
            with self.assertRaises(self.exception_type) as cm:
                self.func(self.input)
            exception = cm.exception
            error_data = getattr(exception, ERROR_DATA)
            self.assertIsInstance(error_data, ErrorData)
            self._test_create_message(error_data)


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


if __name__ == '__main__':
    unittest.main()
