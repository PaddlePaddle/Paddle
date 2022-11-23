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

import numpy
import unittest

import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.jit import declarative

program_translator = ProgramTranslator()


# 1. print VarBase
@declarative
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
@declarative
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
@declarative
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
@declarative
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
@declarative
def dyfunc_print_with_ifelse(x):
    x_v = fluid.dygraph.to_variable(x)
    if len(x_v.shape) > 1:
        print(x_v)
    else:
        print(x_v)


# 6. print mutiple VarBases
@declarative
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
@declarative
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
    print(x_v, y_v)


class TestPrintBase(unittest.TestCase):

    def setUp(self):
        self.input = numpy.ones(5).astype("int32")
        self.place = fluid.CUDAPlace(
            0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
        self.set_test_func()

    def set_test_func(self):
        raise NotImplementedError("Print test should implement set_test_func")

    def _run(self, to_static):
        program_translator.enable(to_static)

        with fluid.dygraph.guard():
            self.dygraph_func(self.input)

    def get_dygraph_output(self):
        self._run(to_static=False)

    def get_static_output(self):
        self._run(to_static=True)


class TestPrintVariable(TestPrintBase):

    def set_test_func(self):
        self.dygraph_func = dyfunc_print_variable

    def test_transformed_static_result(self):
        self.get_dygraph_output()
        self.get_static_output()


class TestPrintNdArray(TestPrintVariable):

    def set_test_func(self):
        self.dygraph_func = dyfunc_print_ndarray


class TestPrintWithFormat(TestPrintVariable):

    def set_test_func(self):
        self.dygraph_func = dyfunc_print_with_format


class TestPrintWithFormat2(TestPrintVariable):

    def set_test_func(self):
        self.dygraph_func = dyfunc_print_with_format2


class TestPrintWithIfElse(TestPrintVariable):

    def set_test_func(self):
        self.dygraph_func = dyfunc_print_with_ifelse


class TestPrintMultipleVar(TestPrintVariable):

    def set_test_func(self):
        self.dygraph_func = dyfunc_print_multi_vars


class TestPrintContinueVar(TestPrintVariable):

    def set_test_func(self):
        self.dygraph_func = dyfunc_print_continue_vars


if __name__ == '__main__':
    unittest.main()
