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

import contextlib
import io
import sys
import six
import numpy
import unittest

import paddle.fluid as fluid
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ProgramTranslator


@signature_safe_contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout


# 1. print VarBase
def dyfunc_print_variable(x):
    x_v = fluid.dygraph.to_variable(x)
    # NOTE: transform to static code, var name will be changed
    x_v.name = "assign_0.tmp_0"
    print(x_v)


# 2. print ndarray
def dyfunc_print_ndarray(x):
    print(x)


# 3. print VarBase with format
def dyfunc_print_with_format(x):
    x_v = fluid.dygraph.to_variable(x)
    print("PrintVariable: {}".format(x_v))


# 4. print VarBase with format 2
def dyfunc_print_with_format2(x):
    x_v = fluid.dygraph.to_variable(x)
    print("PrintVariable: %s" % (x_v))


# 5. print VarBase in control flow1
def dyfunc_print_with_ifelse(x):
    x_v = fluid.dygraph.to_variable(x)
    if len(x_v.shape) > 1:
        print(x_v)
    else:
        print(x_v)


class TestPrintBase(unittest.TestCase):
    def setUp(self):
        self.input = numpy.ones(5).astype("int32")
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.set_test_func()

    def set_test_func(self):
        raise NotImplementedError("Print test should implement set_test_func")

    def get_dygraph_output(self):
        f = io.StringIO()
        with stdout_redirector(f):
            with fluid.dygraph.guard():
                self.dygraph_func(self.input)
        return f.getvalue()

    def get_static_output(self):
        # TODO: How to catch C++ stdout to python
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            program_translator = ProgramTranslator()
            static_func = program_translator.get_func(self.dygraph_func)
            static_func(self.input)

        exe = fluid.Executor(self.place)
        exe.run(main_program)


class TestPrintVariable(TestPrintBase):
    def set_test_func(self):
        self.dygraph_func = dyfunc_print_variable

    def test_transformed_static_result(self):
        print(self.get_dygraph_output())
        self.get_static_output()
        # dy_out = self.get_dygraph_output()
        # static_out = self.get_static_output()
        # self.assertEqual(dy_out, static_out)


class TestPrintNdArray(TestPrintBase):
    def set_test_func(self):
        self.dygraph_func = dyfunc_print_ndarray

    def test_transform_static_error(self):
        with self.assertRaises(TypeError):
            self.get_dygraph_output()
            self.get_static_output()


class TestPrintWithFormat(TestPrintBase):
    def set_test_func(self):
        self.dygraph_func = dyfunc_print_with_format

    def test_transform_static_error(self):
        with self.assertRaises(NotImplementedError):
            self.get_dygraph_output()
            self.get_static_output()


class TestPrintWithFormat2(TestPrintBase):
    def set_test_func(self):
        self.dygraph_func = dyfunc_print_with_format2

    def test_transform_static_error(self):
        with self.assertRaises(NotImplementedError):
            self.get_dygraph_output()
            self.get_static_output()


class TestPrintWithIfElse(TestPrintVariable):
    def set_test_func(self):
        self.dygraph_func = dyfunc_print_with_ifelse


if __name__ == '__main__':
    unittest.main()
