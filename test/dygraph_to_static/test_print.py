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

import unittest

import numpy
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
)

import paddle


# 1. print Tensor
def dyfunc_print_variable(x):
    # NOTE: transform to static code, var name will be changed
    x_t = paddle.to_tensor(x)
    print(x_t)


# 2. print ndarray
def dyfunc_print_ndarray(x):
    print(x)


# 3. print Tensor with format
def dyfunc_print_with_format(x):
    x_t = paddle.to_tensor(x)
    print(f"PrintTensor: {x_t}")


# 4. print Tensor with format 2
def dyfunc_print_with_format2(x):
    x_t = paddle.to_tensor(x)
    print("PrintTensor: %s" % x_t)  # noqa: UP031


# 5. print Tensor in control flow1
def dyfunc_print_with_ifelse(x):
    x_t = paddle.to_tensor(x)
    if len(x_t.shape) > 1:
        print(x_t)
    else:
        print(x_t)


# 6. print multiple Tensor
def dyfunc_print_multi_tensor(x):
    x_t = paddle.to_tensor(x)
    y_t = x_t * 2
    print(x_t)
    print(y_t)


# 7. print continue Tensor
def dyfunc_print_continue_vars(x):
    x_t = paddle.to_tensor(x)
    y_t = x_t * 2
    print(x_t, y_t)


# 8. print with kwargs
def dyfunc_print_with_kwargs(x):
    x_t = paddle.to_tensor(x)
    print("Tensor", x_t, end='\n\n', sep=': ')


class TestPrintBase(Dy2StTestBase):
    def setUp(self):
        self.input = numpy.ones(5).astype("int32")
        self.set_test_func()

    def set_test_func(self):
        raise NotImplementedError("Print test should implement set_test_func")

    def _run(self, to_static: bool):
        with enable_to_static_guard(to_static):
            paddle.jit.to_static(self.dygraph_func)(self.input)

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


class TestPrintMultipleTensor(TestPrintVariable):
    def set_test_func(self):
        self.dygraph_func = dyfunc_print_multi_tensor


class TestPrintContinueVar(TestPrintVariable):
    def set_test_func(self):
        self.dygraph_func = dyfunc_print_continue_vars


class TestPrintWithKwargs(TestPrintVariable):
    def set_test_func(self):
        self.dygraph_func = dyfunc_print_with_kwargs


if __name__ == '__main__':
    unittest.main()
