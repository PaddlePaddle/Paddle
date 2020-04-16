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
import textwrap
import unittest

import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.jit import dygraph_to_static_code

from ifelse_simple_func import dyfunc_with_if_else


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
        answer = get_source_code(StaticCode1.dyfunc_with_if_else)
        code = dygraph_to_static_code(dyfunc_with_if_else)(x_v)
        self.assertEqual(answer, code)

    def test_program_translator(self):
        answer = get_source_code(StaticCode2.dyfunc_with_if_else)
        program_translator = ProgramTranslator()
        code = program_translator.get_code(dyfunc_with_if_else)
        self.assertEqual(answer, code)


if __name__ == '__main__':
    unittest.main()
