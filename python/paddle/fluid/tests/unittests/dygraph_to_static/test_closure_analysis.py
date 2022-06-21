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

import paddle
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.layers.utils import map_structure
from paddle.fluid.dygraph.dygraph_to_static.loop_transformer import NameVisitor
from paddle.utils import gast
import inspect

SEED = 2020
np.random.seed(SEED)


def test_normal_0(x):

    def func():
        if True:
            i = i + 1

    func()
    return i


def test_normal_argument(x):
    x = 1

    def func():
        if True:
            print(x)
            i = i + 1

    func()
    return x


def test_global(x):
    global t
    t = 10

    def func():
        if True:
            print(x)
            i = i + 1

    func()
    return x


def test_nonlocal(x):
    i = 10

    def func():
        nonlocal i
        k = 10
        if True:
            print(x)
            i = i + 1

    func()
    return x


class TestClosureAnalysis(unittest.TestCase):

    def setUp(self):
        self.init_dygraph_func()

    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            test_nonlocal, test_global, test_normal_0, test_normal_argument
        ]
        self.answer = [
            {
                'func': set('k'),
                'test_nonlocal': set('i')
            },
            {
                'func': set('i'),
            },
            {
                'func': set('i'),
            },
            {
                'func': set('i'),
            },
        ]

    def test_main(self):
        for ans, func in zip(self.answer, self.all_dygraph_funcs):
            test_func = inspect.getsource(func)
            gast_root = gast.parse(test_func)
            name_visitor = NameVisitor(gast_root)
            d = name_visitor.func_to_created_variables
            d = {k.name: v for k, v in d.items()}
            self.assertEqual(len(d), len(ans))
            for name in d.keys():
                self.assertEqual(d[name], ans[name])


if __name__ == '__main__':
    unittest.main()
