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
from paddle.fluid.dygraph.jit import dygraph_to_static_code

from ifelse_simple_func import dyfunc_with_if_else


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


if __name__ == '__main__':
    unittest.main()
