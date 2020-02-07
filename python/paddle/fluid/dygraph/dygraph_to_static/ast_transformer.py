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

__all__ = ['DygraphToStaticAst']


class NodeVarType(object):
    """
    Enum class of python variable types. We have to know some variable types
    during compile time to transfer AST. For example, a string variable and a
    tensor variable in if clause may lead to different conversion from dygraph
    to static graph.
    """
    UNKNOWN = 0  # Reserve for AST nodes have not known the type
    STATEMENT = 1  # For nodes representing statement (non-variable type)
    PADDLE_DYGRAPH_API = 2
    PADDLE_CONTROL_IF = 3
    PADDLE_CONTROL_WHILE = 4
    PADDLE_CONTROL_FOR = 5

    NONE = 100
    INT = 101
    FLOAT = 102
    STRING = 103
    TENSOR = 104


class AstNodeWrapper(object):
    """
    Wrapper for python ast.node. We need a node wrapper because ast.node
    doesn't store all required information when we are transforming AST.
    We should collect additional information which the actual transformation
    needs.
    """

    def __init__(self, node):
        self.node = node
        self.parent = None
        self.node_var_type = NodeVarType.UNKNOWN


class DygraphToStaticAst(ast.NodeTransformer):
    """
    Main class to transform Dygraph to Static Graph
    """

    def get_static_ast(self, root):
        # save root for some analysis may need global AST 
        self.root = root
        self.static_analysis_root = AstNodeWrapper(root)
        self.visit(root)
        self.transfer_from_node_type(self.static_analysis_root)
        return self.static_analysis_root

    def visit(self, node):
        # TODO construct a tree whose nodes are AstNodeWrapper
        # This step also does static node type analysis 
        print("Not implemented")

    def transfer_from_node_type(self, node):
        print("Not implemented")
