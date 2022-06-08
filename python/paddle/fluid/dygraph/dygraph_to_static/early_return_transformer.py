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

import six
import copy
from collections import defaultdict

from paddle.utils import gast
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper


class EarlyReturnTransformer(gast.NodeTransformer):
    """
    Transform if/else statement of Dygraph into Static Graph.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Type of input node should be AstNodeWrapper, but received %s ." % type(
            wrapper_root)
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
            node)
        for child in node.body:
            if isinstance(child, gast.Return):
                return True
        return False

    def visit_stmt_block(self, nodes):
        result = []
        node_destination = result
        for node in nodes:
            replacement = self.visit(node)

            if isinstance(replacement, (list, tuple)):
                node_destination.extend(replacement)
            else:
                node_destination.append(replacement)

            if isinstance(node, gast.If) and self.is_define_return_in_if(node):
                node_destination = node.orelse

        return result

    def visit_If(self, node):
        node.body = self.visit_stmt_block(node.body)
        node.orelse = self.visit_stmt_block(node.orelse)
        return node

    def visit_FunctionDef(self, node):
        node.body = self.visit_stmt_block(node.body)
        return node
