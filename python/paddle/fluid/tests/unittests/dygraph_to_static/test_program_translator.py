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

import unittest
import numpy as np

import paddle.fluid as fluid

from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
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


class TestDygraphToStaticCode(unittest.TestCase):
    def setUp(self):
        # set to print all string diff when assertEqual fails
        self.maxDiff = None

    def test_decorator(self):
        answer = "\
def dyfunc_with_if_else(x_v, label=None):\n\
\n\
    def true_fn_0(x_v):\n\
        x_v = x_v - 1\n\
        return x_v\n\
\n\
    def false_fn_0(x_v):\n\
        x_v = x_v + 1\n\
        return x_v\n\
    x_v = fluid.layers.cond(fluid.layers.mean(x_v)[0] > 5, lambda :\n\
        true_fn_0(x_v), lambda : false_fn_0(x_v))\n\
    if label is not None:\n\
        loss = fluid.layers.cross_entropy(x_v, label)\n\
        return loss\n\
    return x_v\n"

        x_v = None
        program_translator = ProgramTranslator()
        code = program_translator.get_code(dyfunc_with_if_else)
        self.assertEqual(answer, code)

    def test_program_translator(self):
        answer = "\
def dyfunc_with_if_else(x_v, label=None):\n\
\n\
    def true_fn_1(x_v):\n\
        x_v = x_v - 1\n\
        return x_v\n\
\n\
    def false_fn_1(x_v):\n\
        x_v = x_v + 1\n\
        return x_v\n\
    x_v = fluid.layers.cond(fluid.layers.mean(x_v)[0] > 5, lambda :\n\
        true_fn_1(x_v), lambda : false_fn_1(x_v))\n\
    if label is not None:\n\
        loss = fluid.layers.cross_entropy(x_v, label)\n\
        return loss\n\
    return x_v\n"

        program_translator = ProgramTranslator()
        code = program_translator.get_code(dyfunc_with_if_else)
        self.assertEqual(answer, code)


class TestEnableDeclarative(unittest.TestCase):
    def test_enable_disable_returns(self):
        x = np.random.randn(30, 10, 32).astype('float32')
        weight = np.random.randn(32, 64).astype('float32')
        program_translator = ProgramTranslator()

        program_translator.enable_declarative_function(True)
        static_output = program_translator.get_output(simple_func, x, weight)

        program_translator.enable_declarative_function(False)
        with fluid.dygraph.guard():
            dygraph_output = program_translator.get_output(simple_func, x,
                                                           weight)
            self.assertTrue(
                np.allclose(
                    static_output.numpy(), dygraph_output.numpy(), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
