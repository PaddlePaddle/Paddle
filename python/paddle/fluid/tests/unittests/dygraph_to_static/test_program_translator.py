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

import astor
import gast
import inspect
import numpy as np
import textwrap
import unittest

import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.dygraph.nn import Linear

from ifelse_simple_func import dyfunc_with_if_else

np.random.seed(0)


# TODO(Aurelius): Currently, `declarative` don't support decorate the function
# that contains layers with initialized operation, like `fc = linear(10, 3)`.
# Because initialized ops will be added into program and be executed many times.
# The parameters are assumed to initialized outside of the function.
def simple_func(x, weight_numpy):
    x = fluid.dygraph.to_variable(x)
    w = fluid.dygraph.to_variable(weight_numpy)
    y = fluid.layers.matmul(x, w)
    z = fluid.layers.mean(y)
    return z


@declarative
def decorated_simple_func(x, weight_numpy):
    x = fluid.dygraph.to_variable(x)
    w = fluid.dygraph.to_variable(weight_numpy)
    y = fluid.layers.matmul(x, w)
    z = fluid.layers.mean(y)
    return z


def get_source_code(func):
    raw_code = inspect.getsource(func)
    code = textwrap.dedent(raw_code)
    root = gast.parse(code)
    source_code = astor.to_source(gast.gast_to_ast(root))
    return source_code


class StaticCode1():
    def dyfunc_with_if_else(x_v, label=None):
        def true_fn_0(x_v):
            x_v = x_v - 1
            return x_v

        def false_fn_0(x_v):
            x_v = x_v + 1
            return x_v

        x_v = fluid.layers.cond(
            fluid.layers.mean(x_v)[0] > 5,
            lambda: fluid.dygraph.dygraph_to_static.convert_call(true_fn_0)(x_v),
            lambda: fluid.dygraph.dygraph_to_static.convert_call(false_fn_0)(x_v)
        )
        if label is not None:
            loss = fluid.layers.cross_entropy(x_v, label)
            return loss
        return x_v


class StaticCode2():
    def dyfunc_with_if_else(x_v, label=None):
        def true_fn_1(x_v):
            x_v = x_v - 1
            return x_v

        def false_fn_1(x_v):
            x_v = x_v + 1
            return x_v

        x_v = fluid.layers.cond(
            fluid.layers.mean(x_v)[0] > 5,
            lambda: fluid.dygraph.dygraph_to_static.convert_call(true_fn_1)(x_v),
            lambda: fluid.dygraph.dygraph_to_static.convert_call(false_fn_1)(x_v)
        )
        if label is not None:
            loss = fluid.layers.cross_entropy(x_v, label)
            return loss
        return x_v


class NetWithError(fluid.dygraph.layers.Layer):
    @declarative
    def forward(self, x):
        linear = fluid.dygraph.Linear(32, 64)
        y = linear(x)
        return y


class TestDygraphToStaticCode(unittest.TestCase):
    def setUp(self):
        # set to print all string diff when assertEqual fails
        self.maxDiff = None

    def test_decorator(self):
        x_v = None
        program_translator = ProgramTranslator()
        code = program_translator.get_code(dyfunc_with_if_else)
        answer = get_source_code(StaticCode1.dyfunc_with_if_else)
        self.assertEqual(answer, code)

    def test_program_translator(self):
        answer = get_source_code(StaticCode2.dyfunc_with_if_else)
        program_translator = ProgramTranslator()
        code = program_translator.get_code(dyfunc_with_if_else)
        self.assertEqual(answer, code)


class TestEnableDeclarative(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(30, 10, 32).astype('float32')
        self.weight = np.random.randn(32, 64).astype('float32')
        self.program_translator = ProgramTranslator()

    def test_raise_error(self):
        with fluid.dygraph.guard():
            self.program_translator.enable(True)
            net = NetWithError()
            with self.assertRaises(ValueError):
                net(fluid.dygraph.to_variable(self.x))

    def test_enable_disable_get_output(self):
        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            static_output = self.program_translator.get_output(
                simple_func, self.x, self.weight)

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_output = self.program_translator.get_output(
                simple_func, self.x, self.weight)
            self.assertTrue(
                np.allclose(
                    static_output.numpy(), dygraph_output.numpy(), atol=1e-4))

    def test_enable_disable_get_func(self):

        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            static_func = self.program_translator.get_func(simple_func)
            self.assertTrue(callable(static_func))
            static_output = static_func(self.x, self.weight)
            self.assertTrue(isinstance(static_output, fluid.Variable))

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_func = self.program_translator.get_func(simple_func)
            self.assertTrue(callable(dygraph_func))
            dygraph_output = dygraph_func(self.x, self.weight)
            self.assertTrue(isinstance(dygraph_output, fluid.core.VarBase))

    def test_enable_disable_get_program(self):

        self.program_translator.enable(True)
        static_output = self.program_translator.get_program(simple_func, self.x,
                                                            self.weight)
        self.assertTrue(isinstance(static_output, tuple))
        self.assertEqual(len(static_output), 4)
        self.assertTrue(isinstance(static_output[0], fluid.Program))
        self.assertTrue(isinstance(static_output[1], fluid.Program))
        # Check all inputs and outputs are Variable
        for var in static_output[2]:
            self.assertTrue(isinstance(var, fluid.Variable))

        for var in static_output[3]:
            self.assertTrue(isinstance(var, fluid.Variable))

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_output = self.program_translator.get_program(
                simple_func, self.x, self.weight)
            self.assertTrue(isinstance(dygraph_output, fluid.core.VarBase))

    def test_enable_disable_declarative(self):

        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            static_output = decorated_simple_func(self.x, self.weight)

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_output = decorated_simple_func(self.x, self.weight)
            self.assertTrue(
                np.allclose(
                    static_output.numpy(), dygraph_output.numpy(), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
