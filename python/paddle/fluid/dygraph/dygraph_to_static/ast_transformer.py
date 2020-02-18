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
# gast is a generic AST to represent Python2 and Python3's Abstract Syntax Tree(AST).
# It provides a compatibility layer between the AST of various Python versions,
# as produced by ast.parse from the standard ast module.
# See details in https://github.com/serge-sans-paille/gast/
from .ast_utils import is_control_flow_if, create_cond_node, transform_if_else

__all__ = ['AstNodeWrapper', 'DygraphToStaticAst', 'StaticAnalysisVisitor']

DECORATOR_NAME = 'dygraph_to_static_output'


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
        self.children = []
        self.node_var_type = NodeVarType.UNKNOWN


class StaticAnalysisVisitor(object):
    """
    A class that does static analysis
    """

    def __init__(self, ast_root=None):
        if ast_root is not None:
            self.run(ast_root)

    def run(self, ast_root):
        self.node_wrapper_root = None
        self.ancestor_wrappers = []
        self.node_to_wrapper_map = {}
        self.dfs_visit(ast_root)

    def dfs_visit(self, node):
        # AST reuses some ast.nodes, such as Param node of expr_context
        if node not in self.node_to_wrapper_map:
            cur_wrapper = AstNodeWrapper(node)
            self.node_to_wrapper_map[node] = cur_wrapper
        else:
            cur_wrapper = self.node_to_wrapper_map[node]

        if self.node_wrapper_root is None:
            self.node_wrapper_root = cur_wrapper

        if len(self.ancestor_wrappers) != 0:
            last_wrapper = self.ancestor_wrappers[-1]
            last_wrapper.children.append(cur_wrapper)
            cur_wrapper.parent = last_wrapper

        self.ancestor_wrappers.append(cur_wrapper)
        for child in gast.iter_child_nodes(node):
            self.dfs_visit(child)
        self.ancestor_wrappers.pop()
        return cur_wrapper.node_var_type

    def get_node_wrapper_root(self):
        return self.node_wrapper_root

    def get_node_to_wrapper_map(self):
        return self.node_to_wrapper_map


class DygraphToStaticAst(gast.NodeTransformer):
    """
    Main class to transform Dygraph to Static Graph
    """

    def get_static_ast(self, root):
        # save root for some analysis may need global AST
        self.root = root
        self.static_analysis_root = StaticAnalysisVisitor(
            root).get_node_wrapper_root()
        self.static_analysis_root = root
        # record all created ast.functionDef in control flow statement
        self.new_func_nodes = []
        self.decorate_func_name = None
        self.transfer_from_node_type(self.static_analysis_root)
        return self.static_analysis_root, self.decorate_func_name

    def transfer_from_node_type(self, node):
        self.visit(node)
        # add new ast.funcDef of `if/else`
        node.body = self.new_func_nodes + node.body

    def visit_If(self, node):
        assert isinstance(node, gast.If)
        self.generic_visit(node)
        if is_control_flow_if(node.test):
            pred_node = node.test
            true_func_node, false_func_node, return_name_ids = transform_if_else(
                node, self.root)
            self.new_func_nodes += [true_func_node, false_func_node]
            # create layers.cond
            new_node = create_cond_node(return_name_ids, pred_node,
                                        true_func_node, false_func_node)
            return new_node
        else:
            return node

    def visit_Call(self, node):
        # Remove `numpy()` statement
        # like `Tensor.numpy()[i]` -> `Tensor[i]`
        if isinstance(node.func, gast.Attribute):
            attribute = node.func
            if attribute.attr == 'numpy':
                node = attribute.value
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        if self.decorate_func_name is None:
            self.decorate_func_name = node.name
        self.generic_visit(node)
        # Remove the decorated name of dygraph_to_static
        if hasattr(node, 'decorator_list'):
            decorator_list = [
                d for d in node.decorator_list if d.id != DECORATOR_NAME
            ]
            node.decorator_list = decorator_list
        return node
