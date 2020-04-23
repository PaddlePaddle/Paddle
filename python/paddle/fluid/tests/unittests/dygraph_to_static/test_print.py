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


# 1. print VarBase
def dyfunc_print_variable(x):
    """
    PY2:
    Print(dest=None, values=[Name(id='x_v', annotation=None, type_comment=None)], nl=True)],
    PY3:
    Expr(
        value=Call(func=Name(id='print', annotation=None, type_comment=None),
            args=[Name(id='x_v', annotation=None, type_comment=None)],
            keywords=[]))
    """
    # NOTE: transform to static code, var name will be changed
    x_v = fluid.dygraph.to_variable(x)
    print(x_v)


# 2. print ndarray
def dyfunc_print_ndarray(x):
    """
    PY2:
    Print(dest=None, values=[Name(id='x', annotation=None, type_comment=None)
    PY3:
    Expr(
        value=Call(func=Name(id='print', annotation=None, type_comment=None),
            args=[Name(id='x', annotation=None, type_comment=None)],
            keywords=[]))
    """
    print(x)


# 3. print VarBase with format


def dyfunc_print_with_format(x):
    """
    PY2:
    Print(dest=None,
        values=[
            Call(
                func=Attribute(value=Constant(value='PrintVariable: {}', kind=None), attr='format'),
                args=[Name(id='x_v', annotation=None, type_comment=None)],
                keywords=[])],
        nl=True)
    PY3:
    Expr(
        value=Call(func=Name(id='print', annotation=None, type_comment=None),
            args=[
                Call(
                    func=Attribute(value=Constant(value='PrintVariable: {}', kind=None), attr='format'),
                    args=[Name(id='x_v', annotation=None, type_comment=None)],
                    keywords=[])],
            keywords=[]))
    """
    x_v = fluid.dygraph.to_variable(x)
    print("PrintVariable: {}".format(x_v))


# 4. print VarBase with format 2
def dyfunc_print_with_format2(x):
    """
    PY2:
    Print(dest=None,
        values=[
            BinOp(left=Constant(value='PrintVariable: %s', kind=None),
                op=Mod,
                right=Name(id='x_v', annotation=None, type_comment=None))],
        nl=True)
    PY3:
    Expr(
        value=Call(func=Name(id='print', annotation=None, type_comment=None),
            args=[
                BinOp(left=Constant(value='PrintVariable: %s', kind=None),
                    op=Mod,
                    right=Name(id='x_v', annotation=None, type_comment=None))],
            keywords=[]))
    """
    x_v = fluid.dygraph.to_variable(x)
    print("PrintVariable: %s" % (x_v))


# 5. print VarBase in control flow1
def dyfunc_print_with_ifelse(x):
    x_v = fluid.dygraph.to_variable(x)
    if len(x_v.shape) > 1:
        print(x_v)
    else:
        print(x_v)


# 6. print mutiple VarBases
def dyfunc_print_multi_vars(x):
    """
    # NOTE: y_v type is error before cur PR in this case
    Assign(targets=[Name(id='y_v', annotation=None, type_comment=None)],
        value=BinOp(left=Name(id='x_v', annotation=None, type_comment=None), op=Mult, right=Constant(value=2, kind=None)))
    """
    x_v = fluid.dygraph.to_variable(x)
    y_v = x_v * 2
    print(x_v)
    print(y_v)


# 7. print continue VarBase
def dyfunc_print_continue_vars(x):
    """
    PY3:
    Expr(
        value=Call(func=Name(id='print', annotation=None, type_comment=None),
            args=[Name(id='x_v', annotation=None, type_comment=None),
                Name(id='y_v', annotation=None, type_comment=None)],
            keywords=[]))
    PY2:
    Print(dest=None,
        values=[
            Tuple(
                elts=[Name(id='x_v', annotation=None, type_comment=None),
                    Name(id='y_v', annotation=None, type_comment=None)])],
        nl=True)
    """
    x_v = fluid.dygraph.to_variable(x)
    y_v = x_v * 2
    z_v = x_v * 3
    print(x_v, y_v, z_v)


class TestPrintBase(unittest.TestCase):
    def setUp(self):
        self.input = numpy.ones(5).astype("int32")
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.set_test_func()

    def set_test_func(self):
        raise NotImplementedError("Print test should implement set_test_func")

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            self.dygraph_func(self.input)

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
        self.get_dygraph_output()
        self.get_static_output()


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


class TestPrintMultipleVar(TestPrintVariable):
    def set_test_func(self):
        self.dygraph_func = dyfunc_print_multi_vars


class TestPrintContinueVar(TestPrintBase):
    def set_test_func(self):
        self.dygraph_func = dyfunc_print_continue_vars

    def test_transform_static_error(self):
        with self.assertRaises(AssertionError):
            self.get_dygraph_output()
            self.get_static_output()


if __name__ == '__main__':
    unittest.main()
