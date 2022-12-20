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

from paddle.utils import gast

from .base_transformer import BaseTransformer
from .static_analysis import AstNodeWrapper

__all__ = []


class EarlyReturnTransformer(BaseTransformer):
    """
    Transform if/else return statement of Dygraph into Static Graph.
    """

    def __init__(self, wrapper_root):
        assert isinstance(wrapper_root, AstNodeWrapper), (
            "Type of input node should be AstNodeWrapper, but received %s ."
            % type(wrapper_root)
        )
        self.root = wrapper_root.node

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)

    def is_define_return_in_if(self, node):
        assert isinstance(
            node, gast.If
        ), "Type of input node should be gast.If, but received %s ." % type(
            node
        )
        for child in node.body:
            if isinstance(child, gast.Return):
                return True
        return False

    def visit_block_nodes(self, nodes):
        result_nodes = []
        destination_nodes = result_nodes
        for node in nodes:
            rewritten_node = self.visit(node)

            if isinstance(rewritten_node, (list, tuple)):
                destination_nodes.extend(rewritten_node)
            else:
                destination_nodes.append(rewritten_node)

            # append other nodes to if.orelse even though if.orelse is not empty
            if isinstance(node, gast.If) and self.is_define_return_in_if(node):
                destination_nodes = node.orelse
                # handle stmt like `if/elif/elif`
                while (
                    len(destination_nodes) > 0
                    and isinstance(destination_nodes[0], gast.If)
                    and self.is_define_return_in_if(destination_nodes[0])
                ):
                    destination_nodes = destination_nodes[0].orelse

        return result_nodes

    def visit_If(self, node):
        node.body = self.visit_block_nodes(node.body)
        node.orelse = self.visit_block_nodes(node.orelse)
        return node

    def visit_While(self, node):
        node.body = self.visit_block_nodes(node.body)
        node.orelse = self.visit_block_nodes(node.orelse)
        return node

    def visit_For(self, node):
        node.body = self.visit_block_nodes(node.body)
        node.orelse = self.visit_block_nodes(node.orelse)
        return node

    def visit_FunctionDef(self, node):
        node.body = self.visit_block_nodes(node.body)
        return node
