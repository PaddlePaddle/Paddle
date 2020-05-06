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
import gast

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import is_paddle_api


class CallTransformer(gast.NodeTransformer):
    """
    This class transforms function calls into Static Graph Ast.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of CallTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def _is_builtin_call(self, node):
        assert isinstance(node, gast.Call)
        func_str = ast_to_source_code(node.func).strip()
        try:
            from paddle.fluid.dygraph.dygraph_to_static.convert_call_func import is_builtin
            return eval("is_builtin({})".format(func_str))
        except Exception:
            return False

    def transform(self):
        self.visit(self.root)

    def visit_Call(self, node):
        self.generic_visit(node)
        if is_paddle_api(node):
            return node

        if self._is_builtin_call(node):
            return node

        func_str = ast_to_source_code(node.func).strip()
        new_func_str = "fluid.dygraph.dygraph_to_static.convert_call({})".format(
            func_str)
        new_func_ast = gast.parse(new_func_str).body[0].value
        node.func = new_func_ast

        return node
