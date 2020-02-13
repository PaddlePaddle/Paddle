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

import ast
import inspect
import unittest

from paddle.fluid.dygraph.dygraph_to_static import AstNodeWrapper, StaticAnalysisVisitor


def func_to_test_1(a, b):
    return a + b


def func_to_test_2(x):
    for i in range(10):
        x += i
    m = 3
    while m < 8:
        m += 1
    if x < 0:
        return 0
    else:
        return x


class TestStaticAnalysis(unittest.TestCase):
    def _check_wrapper(self, wrapper, node_to_wrapper_map):
        self.assertEqual(node_to_wrapper_map[wrapper.node], wrapper)
        if wrapper.parent is not None:
            self.assertTrue(wrapper in wrapper.parent.children)

        children_ast_nodes = [
            child for child in ast.iter_child_nodes(wrapper.node)
        ]
        self.assertEqual(len(wrapper.children), len(children_ast_nodes))
        for child in wrapper.children:
            self.assertTrue(child.node in children_ast_nodes)
            self._check_wrapper(child, node_to_wrapper_map)

    def test_construct_node_wrapper(self):
        for func in [func_to_test_1, func_to_test_2]:
            test_source_code = inspect.getsource(func)
            ast_root = ast.parse(test_source_code)

            visitor = StaticAnalysisVisitor(ast_root)
            wrapper_root = visitor.get_node_wrapper_root()
            node_to_wrapper_map = visitor.get_node_to_wrapper_map()
            self._check_wrapper(wrapper_root, node_to_wrapper_map)


if __name__ == '__main__':
    unittest.main()
