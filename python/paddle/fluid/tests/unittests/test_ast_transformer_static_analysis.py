#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import gast
import inspect
import numpy as np
import paddle.fluid as fluid
import unittest

from paddle.fluid.dygraph.dygraph_to_static import AstNodeWrapper, NodeVarType, StaticAnalysisVisitor


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


result_var_type2 = {'m': NodeVarType.INT}


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


result_var_type3 = {
    'a': NodeVarType.INT,
    'b': NodeVarType.FLOAT,
    'c': NodeVarType.FLOAT,
    'd': NodeVarType.FLOAT,
    'e': NodeVarType.BOOLEAN,
    'f': NodeVarType.INT,
    'g': NodeVarType.STRING,
    'h': NodeVarType.NONE,
    'i': NodeVarType.BOOLEAN,
    'j': NodeVarType.UNKNOWN
}


def func_to_test4():
    with fluid.dygraph.guard():
        a = np.random.uniform(0.1, 1, [1, 2])
        b = 1 + a
        c = fluid.dygraph.to_variable(b)
        d = (c + 1) * 0.3


result_var_type4 = {
    'a': NodeVarType.NUMPY_NDARRAY,
    'b': NodeVarType.NUMPY_NDARRAY,
    'c': NodeVarType.TENSOR,
    'd': NodeVarType.TENSOR
}

test_funcs = [func_to_test1, func_to_test2, func_to_test3, func_to_test4]
result_var_type = [
    result_var_type1, result_var_type2, result_var_type3, result_var_type4
]


class TestStaticAnalysis(unittest.TestCase):
    def _check_wrapper(self, wrapper, node_to_wrapper_map):
        self.assertEqual(node_to_wrapper_map[wrapper.node], wrapper)
        if wrapper.parent is not None:
            self.assertTrue(wrapper in wrapper.parent.children)

        children_ast_nodes = [
            child for child in gast.iter_child_nodes(wrapper.node)
        ]
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

    def test_var_env(self):
        for i in range(4):
            func = test_funcs[i]
            var_type = result_var_type[i]
            test_source_code = inspect.getsource(func)
            ast_root = gast.parse(test_source_code)
            print(gast.dump(ast_root))
            visitor = StaticAnalysisVisitor(ast_root)
            var_env = visitor.get_var_env()
            scope_var_type = var_env.get_scope_var_type()
            self.assertEqual(len(scope_var_type), len(var_type))
            for name in scope_var_type:
                print("Test var name %s" % (name))
                self.assertTrue(name in var_type)
                self.assertEqual(scope_var_type[name], var_type[name])


if __name__ == '__main__':
    unittest.main()
