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

import paddle
from paddle.fluid.dygraph.dygraph_to_static.utils import FunctionNameLivenessAnalysis
from paddle.utils import gast
import inspect
from numpy import append

global_a = []


class JudgeVisitor(gast.NodeVisitor):

    def __init__(self, ans, mod):
        self.ans = ans
        self.mod = mod

    def visit_FunctionDef(self, node):
        scope = node.pd_scope
        expected = self.ans.get(node.name, set())
        exp_mod = self.mod.get(node.name, set())
        assert scope.existed_vars() == expected, "Not Equals."
        assert scope.modified_vars(
        ) == exp_mod, "Not Equals in function:{} . expect {} , but get {}".format(
            node.name, exp_mod, scope.modified_vars())
        self.generic_visit(node)


class JudgePushPopVisitor(gast.NodeVisitor):

    def __init__(self, push_pop_vars):
        self.pp_var = push_pop_vars

    def visit_FunctionDef(self, node):
        scope = node.pd_scope
        expected = self.pp_var.get(node.name, set())
        assert scope.push_pop_vars == expected, "Not Equals in function:{} . expect {} , but get {}".format(
            node.name, expected, scope.push_pop_vars)
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


def test_push_pop_1(x, *args, **kargs):
    """ push_pop_vars in main_function is : `l`, `k`
    """
    l = []
    k = []
    for i in range(10):
        l.append(i)
        k.pop(i)
    return l


def test_push_pop_2(x, *args, **kargs):
    """ push_pop_vars in main_function is : `k`
    """
    l = []
    k = []

    def func():
        l.append(0)

    for i in range(10):
        k.append(i)
    return l, k


def test_push_pop_3(x, *args, **kargs):
    """ push_pop_vars in main_function is : `k`
        NOTE: One may expect `k` and `l` because l
              is nonlocal. Name bind analysis is
              not implemented yet.
    """
    l = []
    k = []

    def func():
        nonlocal l
        l.append(0)

    for i in range(10):
        k.append(i)
    return l, k


def test_push_pop_4(x, *args, **kargs):
    """ push_pop_vars in main_function is : `k`
    """
    l = []
    k = []
    for i in range(10):
        for j in range(10):
            if True:
                l.append(j)
            else:
                k.pop()
    return l, k


class TestClosureAnalysis(unittest.TestCase):

    def setUp(self):
        self.judge_type = "var and w_vars"
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

        self.modified_var = [
            {
                'func': set('ki'),
                'test_nonlocal': set('i')
            },
            {
                'func': set({'i'}),
                'test_global': set({"t"})
            },
            {
                'func': set('i'),
            },
            {
                'func': set('i'),
                'test_normal_argument': set('x')
            },
        ]

    def test_main(self):
        if self.judge_type == 'push_pop_vars':
            for push_pop_vars, func in zip(self.push_pop_vars,
                                           self.all_dygraph_funcs):
                test_func = inspect.getsource(func)
                gast_root = gast.parse(test_func)
                name_visitor = FunctionNameLivenessAnalysis(gast_root)
                JudgePushPopVisitor(push_pop_vars).visit(gast_root)
        else:
            for mod, ans, func in zip(self.modified_var, self.answer,
                                      self.all_dygraph_funcs):
                test_func = inspect.getsource(func)
                gast_root = gast.parse(test_func)
                name_visitor = FunctionNameLivenessAnalysis(gast_root)
                JudgeVisitor(ans, mod).visit(gast_root)


def TestClosureAnalysis_Attribute_func():
    # in this function, only self is a Name, self.current is a Attribute. self is read and self.current.function is store()
    i = 0
    self.current.function = 12


class TestClosureAnalysis_Attribute(TestClosureAnalysis):

    def init_dygraph_func(self):

        self.all_dygraph_funcs = [TestClosureAnalysis_Attribute_func]
        self.answer = [{"TestClosureAnalysis_Attribute_func": set({'i'})}]
        self.modified_var = [{
            "TestClosureAnalysis_Attribute_func":
            set({'i', 'self.current.function'})
        }]


class TestClosureAnalysis_PushPop(TestClosureAnalysis):

    def init_dygraph_func(self):
        self.judge_type = "push_pop_vars"
        self.all_dygraph_funcs = [
            test_push_pop_1, test_push_pop_2, test_push_pop_3, test_push_pop_4
        ]
        self.push_pop_vars = [{
            "test_push_pop_1": set({'l', 'k'}),
        }, {
            "test_push_pop_2": set({'k'}),
            "func": set("l"),
        }, {
            "test_push_pop_3": set({'k'}),
            "func": set("l"),
        }, {
            "test_push_pop_4": set({'k', 'l'}),
        }]


class TestPushPopTrans(unittest.TestCase):

    def test(self):

        def vlist_of_dict(x):
            ma = {'a': []}
            for i in range(3):
                ma['a'].append(1)
            return ma

        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(vlist_of_dict).code)
        print(paddle.jit.to_static(vlist_of_dict)(x))

    def test2(self):
        import numpy as np

        def vlist_of_dict(x):
            a = np.array([1, 2, 3])
            for i in range(3):
                np.append(a, 4)
            return a

        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(vlist_of_dict).code)
        print(paddle.jit.to_static(vlist_of_dict)(x))

    def test3(self):
        import numpy as np

        def vlist_of_dict(x):
            a = np.array([1, 2, 3])
            if True:
                pass
            return a

        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(vlist_of_dict).code)
        print(paddle.jit.to_static(vlist_of_dict)(x))

    def test4(self):

        def vlist_of_dict(x):
            a = np.array([1, 2, 3])
            for i in range(3):
                append(a, 4)
            return a

        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(vlist_of_dict).code)
        print(paddle.jit.to_static(vlist_of_dict)(x))

    def test5(self):

        def vlist_of_dict(x):
            a = np.array([1, 2, 3])
            for i in range(3):
                global_a.append(4)
            return a

        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(vlist_of_dict).code)
        print(paddle.jit.to_static(vlist_of_dict)(x))


if __name__ == '__main__':
    unittest.main()
