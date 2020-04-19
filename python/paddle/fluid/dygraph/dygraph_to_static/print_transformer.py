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
from paddle.fluid.dygraph.dygraph_to_static.utils import to_print_node


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

    def visit_Expr(self, node):
        value_node = node.value
        for child_node in gast.walk(value_node):
            if isinstance(child_node, gast.Call):
                self._visit_Call(child_node)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if node.func.id == 'print':
            if self._without_format(node):
                if self._is_tensor_node(node.args[0]):
                    to_print_node(node)
                else:
                    raise TypeError(
                        "print object type error, only support print Variable now."
                    )
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

    def _without_format(self, node):
        assert isinstance(node, gast.Call)
        if isinstance(node.args[0], gast.Name):
            return True
        return False
