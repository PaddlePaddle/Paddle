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

from .static_analysis import AstNodeWrapper, StaticAnalysisVisitor

__all__ = ['DygraphToStaticAst']

DECORATOR_NAME = 'dygraph_to_static_output'


class IfElseTransformer(gast.NodeTransformer):
    """
    Transform if/else statement of Dygraph into Static Graph.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Type of input node should be AstNodeWrapper, but received %s ." % type(
            wrapper_root)
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.new_func_nodes = []

    def ast_visit(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)
        self.after_visit(self.root)

    def visit_If(self, node):
        assert isinstance(node, gast.If)
        need_transform = is_control_flow_if(node.test)
        self.generic_visit(node)
        if need_transform:
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
        # Remove `numpy()` statement, like `Tensor.numpy()[i]` -> `Tensor[i]`
        # Todo: should be removed. it may be considered as basic api transformation.
        if isinstance(node.func, gast.Attribute):
            attribute = node.func
            if attribute.attr == 'numpy':
                node = attribute.value
        return node

    def after_visit(self, node):
        """
        This function will add some postprocessing operations with node.
        It can be used to add the created `true_fn/false_fn` in front of
        the node.body before they are called in cond layer.
        """
        assert hasattr(node, 'body')
        # add new ast.funcDef of `if/else`
        if self.new_func_nodes:
            node.body = self.new_func_nodes + node.body

    def get_new_func_nodes(self):
        return self.new_func_nodes


class DygraphToStaticAst(gast.NodeTransformer):
    """
    Main class to transform Dygraph to Static Graph
    """

    def get_static_ast(self, root):
        # save root for some analysis may need global AST
        self.root = root
        self.static_analysis_root = StaticAnalysisVisitor(
            root).get_node_wrapper_root()
        self.decorate_func_name = None
        self.transfer_from_node_type(self.static_analysis_root)
        return self.static_analysis_root

    def transfer_from_node_type(self, node):
        # Generic transformation
        self.visit(node.node)
        # Transform all if/else statement of Dygraph into Static Graph.
        IfElseTransformer(node).ast_visit()

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

    def get_module_name(self):
        """
        Return the main function name which will be used as module name
        in ast_to_func.
        """
        # Should consider BaseAPITransformer which add new module name in Yamei's PR.
        assert self.decorate_func_name, "decorate_func_name shall not be None."
        return self.decorate_func_name
