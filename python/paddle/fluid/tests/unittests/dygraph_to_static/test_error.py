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
def func_error_in_runtime(x):
    x = fluid.dygraph.to_variable(x)
    two = fluid.layers.fill_constant(shape=[1], value=2, dtype="int32")
    x = fluid.layers.reshape(x, shape=[1, two])
    return x


@unwrap
@paddle.jit.to_static()
def func_decorated_by_other_1():
    return 1


@paddle.jit.to_static()
@unwrap
def func_decorated_by_other_2():
    return 1


class LayerErrorInCompiletime(fluid.dygraph.Layer):
    def __init__(self, fc_size=20):
        super(LayerErrorInCompiletime, self).__init__()
        self._linear = fluid.dygraph.Linear(fc_size, fc_size)

    @paddle.jit.to_static(
        input_spec=[paddle.static.InputSpec(
            shape=[20, 20], dtype='float32')])
    def forward(self, x):
        y = self._linear(x)
        z = fluid.layers.fill_constant(shape=[1, 2], value=9, dtype="int")
        out = fluid.layers.mean(y[z])
        return out


class TestFlags(unittest.TestCase):
    def setUp(self):
        self.reset_flags_to_default()

    def reset_flags_to_default(self):
        # Reset flags to use defaut value

        # 1. A flag to set whether to open the dygraph2static error reporting module
        os.environ[error.DISABLE_ERROR_ENV_NAME] = str(
            error.DEFAULT_DISABLE_NEW_ERROR)
        disable_error = int(os.getenv(error.DISABLE_ERROR_ENV_NAME, 999))
        self.assertEqual(disable_error, 0)

        # 2. A flag to set whether to display the simplified error stack
        os.environ[error.SIMPLIFY_ERROR_ENV_NAME] = str(
            error.DEFAULT_SIMPLIFY_NEW_ERROR)
        simplify_error = int(os.getenv(error.SIMPLIFY_ERROR_ENV_NAME, 999))
        self.assertEqual(simplify_error, 1)

    def _test_set_flag(self, flag_name, set_value):
        os.environ[flag_name] = str(set_value)
        new_value = int(os.getenv(error.DISABLE_ERROR_ENV_NAME, 999))
        self.assertEqual(new_value, set_value)

    def test_translator_disable_new_error(self):
        self._test_set_flag(error.DISABLE_ERROR_ENV_NAME, 1)

    def test_translator_simplify_new_error(self):
        self._test_set_flag(error.SIMPLIFY_ERROR_ENV_NAME, 0)


class TestErrorBase(unittest.TestCase):
    def setUp(self):
        self.set_input()
        self.set_func()
        self.set_func_call()
        self.filepath = inspect.getfile(unwrap(self.func_call))
        self.set_exception_type()
        self.set_message()
        self.prog_trans = paddle.jit.ProgramTranslator()

    def set_input(self):
        self.input = np.ones([3, 2])

    def set_func(self):
        raise NotImplementedError("Error test should implement set_func")

    def set_func_call(self):
        raise NotImplementedError("Error test should implement set_func_call")

    def set_exception_type(self):
        raise NotImplementedError(
            "Error test should implement set_exception_type")

    def set_message(self):
        raise NotImplementedError("Error test should implement set_message")

    def reset_flags_to_default(self):
        os.environ[error.DISABLE_ERROR_ENV_NAME] = str(
            error.DEFAULT_DISABLE_NEW_ERROR)
        os.environ[error.SIMPLIFY_ERROR_ENV_NAME] = str(
            error.DEFAULT_SIMPLIFY_NEW_ERROR)

    def disable_new_error(self):
        os.environ[error.DISABLE_ERROR_ENV_NAME] = str(
            1 - error.DEFAULT_DISABLE_NEW_ERROR)

    def _test_new_error_message(self, new_exception, disable_new_error=0):
        error_message = str(new_exception)

        if disable_new_error:
            # If disable new error, 'In user code:' should not in error_message.
            self.assertNotIn('In transformed code:', error_message)
        else:
            # 1. 'In user code:' must be in error_message because it indicates that
            #  this is an optimized error message
            self.assertIn('In transformed code:', error_message)

            # 2. Check whether the converted static graph code is mapped to the dygraph code.
            for m in self.expected_message:
                self.assertIn(m, error_message)

    def _test_raise_new_exception(self, disable_new_error=0):
        paddle.disable_static()

        if disable_new_error:
            self.disable_new_error()
        else:
            self.reset_flags_to_default()

        # 1. Check whether the new exception type is the same as the old one
        with self.assertRaises(self.exception_type) as new_cm:
            self.func_call()

        new_exception = new_cm.exception

        # 2. Check whether the new_exception is attached ErrorData to indicate that this is a new exception
        error_data = getattr(new_exception, error.ERROR_DATA, None)
        self.assertIsInstance(error_data, error.ErrorData)

        # 3. Check whether the error message is optimized
        self._test_new_error_message(new_exception, disable_new_error)


# Situation 1: Call StaticLayer.__call__ to use Dynamic-to-Static
class TestErrorStaticLayerCallInCompiletime(TestErrorBase):
    def set_func(self):
        self.func = func_error_in_compile_time

    def set_input(self):
        self.input = np.ones([3, 2])

    def set_exception_type(self):
        self.exception_type = TypeError

    def set_message(self):
        self.expected_message = \
            ['File "{}", line 35, in func_error_in_compile_time'.format(self.filepath),
             'inner_func()',
             'File "{}", line 28, in inner_func'.format(self.filepath),
             'fluid.layers.fill_constant(shape=[1, 2], value=9, dtype="int")',
             ]

    def set_func_call(self):
        # NOTE: self.func(self.input) is the StaticLayer().__call__(self.input)
        self.func_call = lambda: self.func(self.input)

    def test_error(self):
        for disable_new_error in [0, 1]:
            self._test_raise_new_exception(disable_new_error)


class TestErrorStaticLayerCallInCompiletime_2(
        TestErrorStaticLayerCallInCompiletime):
    def set_func(self):
        self.func = func_error_in_compile_time_2

    def set_exception_type(self):
        self.exception_type = ValueError

    def set_message(self):
        self.expected_message = \
            [
             'File "{}", line 46, in func_error_in_compile_time_2'.format(self.filepath),
             'x = fluid.layers.reshape(x, shape=[1, 2])'
             ]


class TestErrorStaticLayerCallInRuntime(TestErrorStaticLayerCallInCompiletime):
    def set_func(self):
        self.func = func_error_in_runtime

    def set_exception_type(self):
        self.exception_type = ValueError

    def set_message(self):
        self.expected_message = \
            [
                'File "{}", line 54, in func_error_in_runtime'.format(self.filepath),
                'x = fluid.layers.reshape(x, shape=[1, two])'
            ]


# Situation 2: Call ProgramTranslator().get_output(...) to use Dynamic-to-Static
class TestErrorGetOutputInCompiletime(TestErrorStaticLayerCallInCompiletime):
    def set_func_call(self):
        self.func_call = lambda : self.prog_trans.get_output(unwrap(self.func), self.input)


class TestErrorGetOutputInCompiletime_2(
        TestErrorStaticLayerCallInCompiletime_2):
    def set_func_call(self):
        self.func_call = lambda : self.prog_trans.get_output(unwrap(self.func), self.input)


class TestErrorGetOutputInRuntime(TestErrorStaticLayerCallInRuntime):
    def set_func_call(self):
        self.func_call = lambda : self.prog_trans.get_output(unwrap(self.func), self.input)


class TestJitSaveInCompiletime(TestErrorBase):
    def setUp(self):
        self.reset_flags_to_default()
        self.set_func_call()
        self.filepath = inspect.getfile(unwrap(self.func_call))
        self.set_exception_type()
        self.set_message()

    def set_exception_type(self):
        self.exception_type = TypeError

    def set_message(self):
        self.expected_message = \
            ['File "{}", line 80, in forward'.format(self.filepath),
             'fluid.layers.fill_constant(shape=[1, 2], value=9, dtype="int")',
             ]

    def set_func_call(self):
        layer = LayerErrorInCompiletime()
        self.func_call = lambda : paddle.jit.save(layer, path="./test_dy2stat_error/model")

    def test_error(self):
        self._test_raise_new_exception()


# Situation 4: NotImplementedError
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
