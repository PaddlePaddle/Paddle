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

import inspect
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.jit.dy2static import StaticAnalysisVisitor
from paddle.utils import gast


def func_to_test1(a, b):
    return a + b


result_var_type1 = {}


def func_to_test2(x):
    for i in range(10):
        x += i
    m = 3
    while m < 8:
        m += 1
    if x < 0:
        return 0
    else:
        return x


result_var_type2 = {'m': {"INT"}}


def func_to_test3():
    a = 1
    b = 3.0
    c = a * b
    d = True + c
    e = a < b
    f = 9 * (a * 4)
    g = "dddy"
    h = None
    i = False
    j = None + 1
    k: float = 1.0
    l: paddle.Tensor = paddle.to_tensor([1, 2])


result_var_type3 = {
    'a': {"INT"},
    'b': {"FLOAT"},
    'c': {"FLOAT"},
    'd': {"FLOAT"},
    'e': {"BOOLEAN"},
    'f': {"INT"},
    'g': {"STRING"},
    'h': {"NONE"},
    'i': {"BOOLEAN"},
    'j': {"UNKNOWN"},
    'k': {"FLOAT"},
    'l': {"PADDLE_RETURN_TYPES"},
}


def func_to_test4():
    with base.dygraph.guard():
        a = np.random.uniform(0.1, 1, [1, 2])
        b = 1 + a
        c = base.dygraph.to_variable(b)
        d = (c + 1) * 0.3


result_var_type4 = {
    'a': {"NUMPY_NDARRAY"},
    'b': {"NUMPY_NDARRAY"},
    'c': {"TENSOR"},
    'd': {"TENSOR"},
}


def func_to_test5():
    def inner_int_func():
        return 1

    def inner_bool_float_func(x):
        a = 1.0
        if x > 0:
            return a
        return False

    def inner_unknown_func(x):
        return x

    a = inner_int_func()
    b = inner_bool_float_func(3)
    c = inner_unknown_func(None)
    d = paddle.static.data('x', [1, 2])


result_var_type5 = {
    'a': {"INT"},
    'b': {"FLOAT", "BOOLEAN"},
    'c': {"UNKNOWN"},
    'd': {"PADDLE_RETURN_TYPES"},
    'inner_int_func': {"INT"},
    'inner_bool_float_func': {"FLOAT", "BOOLEAN"},
    'inner_unknown_func': {"UNKNOWN"},
}


def func_to_test6(x, y=1):
    i = base.dygraph.to_variable(x)

    def add(x, y):
        return x + y

    while x < 10:
        i = add(i, x)
        x = x + y

    return i


result_var_type6 = {
    'i': {"INT"},
    'x': {"INT"},
    'y': {"INT"},
    'add': {"INT"},
}


def func_to_test7(a: int, b: float, c: paddle.Tensor, d: float = 'diff'):
    a = True
    e, f = paddle.shape(c)
    g: paddle.Tensor = len(c)


result_var_type7 = {
    'a': {"BOOLEAN"},
    'b': {"FLOAT"},
    'c': {"TENSOR"},
    'd': {"STRING"},
    'e': {"PADDLE_RETURN_TYPES"},
    'f': {"PADDLE_RETURN_TYPES"},
    'g': {"TENSOR"},
}

test_funcs = [
    func_to_test1,
    func_to_test2,
    func_to_test3,
    func_to_test4,
    func_to_test5,
    func_to_test6,
    func_to_test7,
]
result_var_type = [
    result_var_type1,
    result_var_type2,
    result_var_type3,
    result_var_type4,
    result_var_type5,
    result_var_type6,
    result_var_type7,
]


class TestStaticAnalysis(unittest.TestCase):
    def _check_wrapper(self, wrapper, node_to_wrapper_map):
        self.assertEqual(node_to_wrapper_map[wrapper.node], wrapper)
        if wrapper.parent is not None:
            self.assertTrue(wrapper in wrapper.parent.children)

        children_ast_nodes = list(gast.iter_child_nodes(wrapper.node))
        self.assertEqual(len(wrapper.children), len(children_ast_nodes))
        for child in wrapper.children:
            self.assertTrue(child.node in children_ast_nodes)
            self._check_wrapper(child, node_to_wrapper_map)

    def test_construct_node_wrapper(self):
        for func in test_funcs:
            test_source_code = inspect.getsource(func)
            ast_root = gast.parse(test_source_code)
            visitor = StaticAnalysisVisitor(ast_root)
            wrapper_root = visitor.get_node_wrapper_root()
            node_to_wrapper_map = visitor.get_node_to_wrapper_map()
            self._check_wrapper(wrapper_root, node_to_wrapper_map)


if __name__ == '__main__':
    unittest.main()
