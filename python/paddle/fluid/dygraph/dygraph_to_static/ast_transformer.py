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
import astor
from .utils import *
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
        # TODO: should be removed. it may be considered as basic api transformation.
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

        # Transform basic api of dygraph to static graph
        BasicApiTransformer(node).ast_visit()

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


class BasicApiTransformer(gast.NodeTransformer):
    """
    Class to transform basic API from dygraph to static graph.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of BasicApiTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.class_node_dict = {}

    def ast_visit(self):
        self.visit(self.root)
        return self.wrapper_root

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

        func_name = astor.to_source(node.func)
        if self._is_dygraph_forward(func_name):
            class_node = self._get_class_node(func_name)
            static_node = to_static_ast(node, class_node)
            return static_node
        else:
            return node

    def _is_dygraph_forward(self, func_id):
        return func_id in self.class_node_dict

    def _get_class_node(self, func_id):
        return self.class_node_dict[func_id]

    def _update_class_node_dict(self, node):
        assert isinstance(node, gast.Assign)
        node_value = node.value
        if isinstance(node_value, gast.Call):
            if is_to_variable(node_value):
                return False

            if is_dygraph_api(node_value):
                update_args_of_func(node_value, node_value, "__init__")
                target_str = astor.to_source(node.targets[0])
                self.class_node_dict[target_str] = node_value
                return True
            # TODO: node.value is not dygraph class
        return False
