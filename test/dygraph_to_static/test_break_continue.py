# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from dygraph_to_static_util import ast_only_test, dy2static_unittest

import paddle
from paddle import base
from paddle.jit.api import to_static
from paddle.jit.dy2static.utils import Dygraph2StaticException

SEED = 2020
np.random.seed(SEED)


@dy2static_unittest
class TestDy2staticException(unittest.TestCase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = None
        self.error = "Your if/else have different number of return value."

    @ast_only_test
    def test_error(self):
        if self.dyfunc:
            with self.assertRaisesRegex(Dygraph2StaticException, self.error):
                paddle.jit.enable_to_static(True)
                self.assertTrue(to_static(self.dyfunc)(self.x))
        paddle.base.dygraph.base.global_var._in_declarative_mode_ = False
        paddle.jit.enable_to_static(False)


def test_continue_in_for(x):
    x = base.dygraph.to_variable(x)
    for i in range(10):
        x += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x


def test_continue_in_for_at_end(x):
    x = base.dygraph.to_variable(x)
    for i in range(10):
        x += 1
        if i > 5:
            continue
    return x


def test_continue_in_while(x):
    x = base.dygraph.to_variable(x)
    i = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=0)
    while i < 10:
        i += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x


def test_break_in_for(x):
    x = base.dygraph.to_variable(x)
    for i in range(10):
        x += 1
        if i > 5:
            break
            x += 10086
        x += i
    return x


def test_break_in_for_at_end(x):
    x = base.dygraph.to_variable(x)
    for i in range(10):
        x += 1
        if i > 5:
            break
    return x


def test_break_in_while(x):
    x = base.dygraph.to_variable(x)
    i = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=0)
    while i < 10:
        i += 1
        if i > 5:
            break
            x += 10086
        x += i
    return x


def test_break_continue_in_for(x):
    x = base.dygraph.to_variable(x)

    for i in range(1, 10, 1):
        if i <= 4:
            x += 1
            continue
        else:
            x += 10010
            break
        x += 10086

    a = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=0)
    b = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=3)
    # b = 10
    # TODO: add Raise Error and suggestion for usage:
    #   Py for contains break/continue depends on control-flow.
    for i in range(b):
        if a <= 4:
            x += 1
            a += 1
            continue
        else:
            x += 10010
            break
        x += 10086

    return x


def test_for_in_else(x):
    x = base.dygraph.to_variable(x)

    # Case 1:
    if False:
        pass
    else:
        for i in range(0, 10):
            if i > 5:
                x += 1
                break
            x += i

    # Case 2:
    if False:
        pass
    else:
        for i in range(0, 10):
            x += 1
            break
            x += i
    return x


def while_loop_class_var(x):
    class Foo:
        def __init__(self):
            self.a = 3
            self.b = 4
            self.c = 5

    foo = Foo()
    i = base.dygraph.to_variable(x)
    while i < 10:
        foo.b = paddle.zeros(shape=[1], dtype='float32')
        foo.c = foo.b + foo.a
        i += 1
        if foo.c < 0:
            continue
        if foo.c > 6:
            break
    return foo.c


def test_optim_break_in_for(x):
    x = paddle.to_tensor(x)
    for i in range(10):
        if x.sum() > 5:
            break
            x += 10086
        x += i
        if i < 3:
            x = x * 2
    return x


def test_optim_break_in_while(x):
    x = paddle.to_tensor(x)
    i = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=0)
    while i < 10:
        if i > 5:
            break
            x += 10086
        x += i
        i += 1
    return x


class TestContinueInFor(unittest.TestCase):
    def setUp(self):
        self.input = np.zeros(1).astype('int64')
        self.place = (
            base.CUDAPlace(0)
            if base.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        self.init_dygraph_func()

    def init_dygraph_func(self):
        self.dygraph_func = test_continue_in_for

    def run_dygraph_mode(self):
        with base.dygraph.guard():
            res = self.dygraph_func(self.input)
            return res.numpy()

    def run_static_mode(self):
        with base.dygraph.guard():
            res = to_static(self.dygraph_func)(self.input)
            return res.numpy()

    def test_transformed_static_result(self):
        static_res = self.run_static_mode()
        dygraph_res = self.run_dygraph_mode()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg='dygraph res is {}\nstatic_res is {}'.format(
                dygraph_res, static_res
            ),
        )


class TestContinueInForAtEnd(TestContinueInFor):
    def init_dygraph_func(self):
        self.dygraph_func = test_continue_in_for_at_end


class TestBreakInFor(TestContinueInFor):
    def init_dygraph_func(self):
        self.dygraph_func = test_break_in_for


class TestBreakInForAtEnd(TestContinueInFor):
    def init_dygraph_func(self):
        self.dygraph_func = test_break_in_for_at_end


class TestBreakContinueInFor(TestContinueInFor):
    def init_dygraph_func(self):
        self.dygraph_func = test_break_continue_in_for


class TestForInElse(TestContinueInFor):
    def init_dygraph_func(self):
        self.dygraph_func = test_for_in_else


class TestContinueInWhile(TestContinueInFor):
    def init_dygraph_func(self):
        self.dygraph_func = test_continue_in_while


class TestBreakInWhile(TestContinueInWhile):
    def init_dygraph_func(self):
        self.dygraph_func = test_break_in_while


class TestWhileLoopClassVar(TestContinueInWhile):
    def init_dygraph_func(self):
        self.dygraph_func = while_loop_class_var


class TestOptimBreakInFor(TestDy2staticException):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = test_optim_break_in_for
        self.error = "python while pred change from bool to variable."


class TestOptimBreakInWhile(TestContinueInWhile):
    def init_dygraph_func(self):
        self.dygraph_func = test_optim_break_in_while


if __name__ == '__main__':
    unittest.main()
