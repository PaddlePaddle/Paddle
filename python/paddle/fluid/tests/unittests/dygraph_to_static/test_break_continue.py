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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import declarative

SEED = 2020
np.random.seed(SEED)


def test_continue_in_for(x):
    x = fluid.dygraph.to_variable(x)
    for i in range(10):
        x += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x


def test_continue_in_for_at_end(x):
    x = fluid.dygraph.to_variable(x)
    for i in range(10):
        x += 1
        if i > 5:
            continue
    return x


def test_continue_in_while(x):
    x = fluid.dygraph.to_variable(x)
    i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    while i < 10:
        i += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x


def test_break_in_for(x):
    x = fluid.dygraph.to_variable(x)
    for i in range(10):
        x += 1
        if i > 5:
            break
            x += 10086
        x += i
    return x


def test_break_in_for_at_end(x):
    x = fluid.dygraph.to_variable(x)
    for i in range(10):
        x += 1
        if i > 5:
            break
    return x


def test_break_in_while(x):
    x = fluid.dygraph.to_variable(x)
    i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    while i < 10:
        i += 1
        if i > 5:
            break
            x += 10086
        x += i
    return x


def test_break_continue_in_for(x):
    x = fluid.dygraph.to_variable(x)
    for i in range(1, 10, 1):
        if i <= 4:
            x += 1
            continue
        else:
            x += 10010
            break
        x += 10086
    return x


def test_for_in_else(x):
    x = fluid.dygraph.to_variable(x)

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
    class Foo(object):
        def __init__(self):
            self.a = 3
            self.b = 4
            self.c = 5

    foo = Foo()
    i = fluid.dygraph.to_variable(x)
    while i < 10:
        foo.b = fluid.layers.zeros(shape=[1], dtype='float32')
        foo.c = foo.b + foo.a
        i += 1
        if foo.c < 0:
            continue
        if foo.c > 6:
            break
    return foo.c


class TestContinueInFor(unittest.TestCase):
    def setUp(self):
        self.input = np.zeros((1)).astype('int32')
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.init_dygraph_func()

    def init_dygraph_func(self):
        self.dygraph_func = test_continue_in_for

    def run_dygraph_mode(self):
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input)
            return res.numpy()

    def run_static_mode(self):
        with fluid.dygraph.guard():
            res = declarative(self.dygraph_func)(self.input)
            return res.numpy()

    def test_transformed_static_result(self):
        static_res = self.run_static_mode()
        dygraph_res = self.run_dygraph_mode()
        self.assertTrue(
            np.allclose(dygraph_res, static_res),
            msg='dygraph res is {}\nstatic_res is {}'.format(dygraph_res,
                                                             static_res))


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


if __name__ == '__main__':
    unittest.main()
