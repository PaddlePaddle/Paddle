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
import astor
from .utils import *

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
        self.class_node_dict = {}
        self.static_analysis_root = StaticAnalysisVisitor(
            root).get_node_wrapper_root()

        # self.transfer_from_node_type(self.static_analysis_root)
        # Just modify self.root now. todo: modify self.static_analysis_root
        self.transfer_from_node_type(self.root)
        return self.root, root.body[0].name

    def transfer_from_node_type(self, node):
        self.visit(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if hasattr(node, 'decorator_list'):
            decorator_list = [
                d for d in node.decorator_list if d.id != DECORATOR_NAME
            ]
            node.decorator_list = decorator_list

        new_node = func_node_from_class(node)
        return new_node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if hasattr(node, 'decorator_list'):
            decorator_list = [
                d for d in node.decorator_list if d.id != DECORATOR_NAME
            ]
            node.decorator_list = decorator_list
        return node

    def visit_Assign(self, node):
        if self._update_class_node_dict(node):
            return None

        value_node = node.value
        for child_node in gast.walk(value_node):
            if isinstance(child_node, gast.Call):
                self._visit_Call(child_node)

        return node

    def visit_Expr(self, node):
        value_node = node.value
        for child_node in gast.walk(value_node):
            if isinstance(child_node, gast.Call):
                if is_dygraph_api(child_node):
                    return
                else:
                    self._visit_Call(child_node)

        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)

        # Replace API `to_variable` with `fluid.layers.assign`
        if is_to_variable(node):
            node = to_assign_node(node)
            return node

        func_id = astor.to_source(node.func)
        if self._is_dygraph_forward(func_id):
            class_node = self._get_class_node(func_id)
            paddle_class, paddle_args, paddle_keywords = parse_class(class_node)
            static_node = to_static_ast(node, paddle_class, paddle_args,
                                        paddle_keywords)
            return static_node
        else:
            return node

    def _is_dygraph_forward(self, func_id):
        return func_id in self.class_node_dict

    def _get_class_node(self, func_id):
        return self.class_node_dict[func_id]

    def _update_class_node_dict(self, node):
        assert isinstance(node, gast.Assign)
        if isinstance(node.value, gast.Call):
            if is_to_variable(node.value):
                return False

            if is_dygraph_api(node.value):
                target_str = astor.to_source(node.targets[0])
                self.class_node_dict[target_str] = node.value
                return True
        return False
