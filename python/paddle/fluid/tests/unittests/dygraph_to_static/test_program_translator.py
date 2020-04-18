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


def simple_func(x, weight_numpy):
    weight_initalizer = fluid.initializer.NumpyArrayInitializer(weight_numpy)
    linear = Linear(32, 64, param_attr=weight_initalizer, bias_attr=False)
    x = fluid.dygraph.to_variable(x)
    y = linear(x)
    z = linear(x)
    return z


@declarative
def decorated_simple_func(x, weight_numpy):
    weight_initalizer = fluid.initializer.NumpyArrayInitializer(weight_numpy)
    linear = Linear(32, 64, param_attr=weight_initalizer, bias_attr=False)
    x = fluid.dygraph.to_variable(x)
    y = linear(x)
    z = linear(x)
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
            fluid.layers.mean(x_v)[0] > 5, lambda: true_fn_0(x_v),
            lambda: false_fn_0(x_v))
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
            fluid.layers.mean(x_v)[0] > 5, lambda: true_fn_1(x_v),
            lambda: false_fn_1(x_v))

        if label is not None:
            loss = fluid.layers.cross_entropy(x_v, label)
            return loss
        return x_v


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
    def test_enable_disable_get_output(self):
        x = np.random.randn(30, 10, 32).astype('float32')
        weight = np.random.randn(32, 64).astype('float32')
        program_translator = ProgramTranslator()

        with fluid.program_guard(fluid.Program(), fluid.Program()):
            program_translator.enable_declarative_function(True)
            static_output = program_translator.get_output(simple_func, x,
                                                          weight)

        program_translator.enable_declarative_function(False)
        with fluid.dygraph.guard():
            dygraph_output = program_translator.get_output(simple_func, x,
                                                           weight)
            self.assertTrue(
                np.allclose(
                    static_output.numpy(), dygraph_output.numpy(), atol=1e-4))

    def test_enable_disable_get_func(self):
        x = np.random.randn(30, 10, 32).astype('float32')
        weight = np.random.randn(32, 64).astype('float32')
        program_translator = ProgramTranslator()

        with fluid.program_guard(fluid.Program(), fluid.Program()):
            program_translator.enable_declarative_function(True)
            static_func = program_translator.get_func(simple_func)
            self.assertTrue(callable(static_func))
            static_output = static_func(x, weight)
            self.assertTrue(isinstance(static_output, fluid.Variable))

        program_translator.enable_declarative_function(False)
        with fluid.dygraph.guard():
            dygraph_func = program_translator.get_func(simple_func)
            self.assertTrue(callable(dygraph_func))
            dygraph_output = dygraph_func(x, weight)
            self.assertTrue(isinstance(dygraph_output, fluid.core.VarBase))

    def test_enable_disable_get_program(self):
        x = np.random.randn(30, 10, 32).astype('float32')
        weight = np.random.randn(32, 64).astype('float32')
        program_translator = ProgramTranslator()

        with fluid.program_guard(fluid.Program(), fluid.Program()):
            program_translator.enable_declarative_function(True)
            static_output = program_translator.get_program(simple_func, x,
                                                           weight)
            self.assertTrue(isinstance(static_output, tuple))
            self.assertEqual(len(static_output), 4)
            self.assertTrue(isinstance(static_output[0], fluid.Program))
            self.assertTrue(isinstance(static_output[1], fluid.Program))

        program_translator.enable_declarative_function(False)
        with fluid.dygraph.guard():
            dygraph_output = program_translator.get_program(simple_func, x,
                                                            weight)
            self.assertTrue(isinstance(dygraph_output, fluid.core.VarBase))

    def test_enable_disable_declarative(self):
        x = np.random.randn(30, 10, 32).astype('float32')
        weight = np.random.randn(32, 64).astype('float32')
        program_translator = ProgramTranslator()

        with fluid.program_guard(fluid.Program(), fluid.Program()):
            program_translator.enable_declarative_function(True)
            static_output = decorated_simple_func(x, weight)

        program_translator.enable_declarative_function(False)
        with fluid.dygraph.guard():
            dygraph_output = decorated_simple_func(x, weight)
            self.assertTrue(
                np.allclose(
                    static_output.numpy(), dygraph_output.numpy(), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
