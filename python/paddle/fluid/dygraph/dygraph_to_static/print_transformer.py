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
import astor

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper, NodeVarType, StaticAnalysisVisitor


class PrintTransformer(gast.NodeTransformer):
    """
    This class transform python print function to fluid.layers.Print.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of PrintTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )

    def transform(self):
        self.visit(self.root)

    # NOTE: deal with print in PY3
    def visit_Expr(self, node):
        if isinstance(node.value, gast.Call):
            node.value = self._visit_Call(node.value)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if isinstance(node.func, gast.Name) and node.func.id == 'print':
            var = self._get_print_var(node)
            return self._contruct_print_node(var)
        return node

    # NOTE: deal with print in PY2
    def visit_Print(self, node):
        var = self._get_print_var(node)
        print_call_node = self._contruct_print_node(var)
        return gast.Expr(value=print_call_node)

    def _get_print_var(self, node):
        if isinstance(node, gast.Call):
            var_list = node.args
        elif isinstance(node, gast.Print):
            var_list = node.values
            if isinstance(var_list[0], gast.Tuple):
                var_list = var_list[0].elts
        # TODO: support print multiple Var
        assert len(var_list) == 1, "Now only support print one Variable."
        return var_list[0]

    def _contruct_print_node(self, node):
        if isinstance(node, gast.Name):
            if self._is_tensor_node(node):
                print_node = gast.Call(
                    func=gast.parse('fluid.layers.Print').body[0].value,
                    args=[node],
                    keywords=[])
                return print_node
            else:
                raise TypeError(
                    "print object type error, only support print Variable now.")
        else:
            # TODO: may not only print with format
            raise NotImplementedError(
                "cannot transform print with format temporarily.")

    def _is_tensor_node(self, node):
        tensor_types = {NodeVarType.TENSOR, NodeVarType.PADDLE_RETURN_TYPES}
        wrapper_node = self.node_to_wrapper_map.get(node, None)
        if wrapper_node is not None:
            if wrapper_node.node_var_type & tensor_types:
                return True
        return False
