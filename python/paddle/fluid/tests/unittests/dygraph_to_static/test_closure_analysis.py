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
from paddle.fluid.dygraph.dygraph_to_static.loop_transformer import FunctionNameLivenessAnalysis
from paddle.utils import gast
import inspect


class JudgeVisitor(gast.NodeVisitor):

    def __init__(self, ans):
        self.ans = ans

    def visit_FunctionDef(self, node):
        scope = node.pd_scope
        expected = self.ans.get(node.name, set())
        assert scope.created_vars() == expected, "Not Equals."
        self.generic_visit(node)


def test_normal_0(x):

    def func():
        if True:
            i = 1

    func()
    return i


def test_normal_argument(x):
    x = 1

    def func():
        if True:
            print(x)
            i = 1

    func()
    return x


def test_global(x):
    global t
    t = 10

    def func():
        if True:
            print(x)
            i = 1

    func()
    return x


def test_nonlocal(x, *args, **kargs):
    i = 10

    def func(*args, **kargs):
        nonlocal i
        k = 10
        if True:
            print(x)
            i = 1

    func(*args, **kargs)
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
                'func': set({'i'}),
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
            name_visitor = FunctionNameLivenessAnalysis(gast_root)
            JudgeVisitor(ans).visit(gast_root)


def TestClosureAnalysis_Attribute_func():
    # in this function, only self is a Name, self.current is a Attribute. self is read and self.current.function is store()
    i = 0
    self.current.function = 12


class TestClosureAnalysis_Attribute(TestClosureAnalysis):

    def init_dygraph_func(self):

        self.all_dygraph_funcs = [TestClosureAnalysis_Attribute_func]
        self.answer = [{"TestClosureAnalysis_Attribute_func": set({'i'})}]


if __name__ == '__main__':
    unittest.main()
