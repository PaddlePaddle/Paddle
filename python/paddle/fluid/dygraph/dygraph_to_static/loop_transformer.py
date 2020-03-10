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

import copy
import gast

from collections import defaultdict
from paddle.fluid import unique_name
from paddle.fluid.dygraph.dygraph_to_static.utils import generate_name_node
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import create_static_variable_gast_node
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import to_static_variable_gast_node

__all__ = ['LoopTransformer', 'NameVisitor']

WHILE_CONDITION_PREFIX = 'while_condition'
WHILE_BODY_PREFIX = 'while_body'


def create_while_node(condition_name, body_name, loop_var_names):
    while_args = []
    while_args.append(
        gast.Name(
            id=condition_name,
            ctx=gast.Param(),
            annotation=None,
            type_comment=None))
    while_args.append(
        gast.Name(
            id=body_name, ctx=gast.Param(), annotation=None, type_comment=None))
    assign_targets = [
        gast.Name(
            id=var_name, ctx=gast.Param(), annotation=None, type_comment=None)
        for var_name in loop_var_names
    ]
    while_args.append(gast.List(elts=assign_targets, ctx=gast.Param()))

    while_func_id = gast.parse('fluid.layers.while_loop').body[0].value
    while_node = gast.Call(func=while_func_id, args=while_args, keywords=[])
    assign_node = gast.Assign(
        targets=[gast.Tuple(
            elts=assign_targets, ctx=gast.Store())],
        value=while_node)
    return assign_node


class NameVisitor(gast.NodeVisitor):
    '''
    Analysis name liveness for loop transformer
    '''

    def __init__(self, root_node):
        # Set of gast.Name
        self.current_seen_vars = set()
        # List of gast.While/gast.For nodes
        self.current_loop = []

        # Mapping from gast.While/gast.For to string name of vars
        self.before_loop_vars = defaultdict(set)
        self.in_loop_vars = defaultdict(set)

        self.visit(root_node)

    def is_control_flow_loop(self, node):
        # TODO: make a better condition
        return True

    def get_loop_var_names(self, node):
        assert isinstance(node, (gast.While,
                                 gast.For)), "Input node is not gast loop node"
        loop_var_names = set()
        create_var_names = set()
        read_context = {type(gast.Load()), type(gast.AugLoad())}

        in_loop_vars = self.in_loop_vars[node]
        in_loop_name_strs = set(name.id for name in in_loop_vars)
        before_loop_vars = self.before_loop_vars[node]
        before_loop_name_strs = set(name.id for name in before_loop_vars)
        after_loop_vars = self.current_seen_vars - before_loop_vars - in_loop_vars
        after_loop_name_strs = set(
            name.id for name in after_loop_vars
            if type(name.ctx) in read_context)
        for name in in_loop_name_strs:
            if name in before_loop_name_strs:
                # If a variable is used in loop and created before loop, it
                # should be in loop_var as input
                loop_var_names.add(name)
            elif name in after_loop_name_strs:
                # If a variable is created in the while loop and read after
                # loop, it should be in loop_var and we should create it
                loop_var_names.add(name)
                create_var_names.add(name)
        return loop_var_names, create_var_names

    def visit_Name(self, node):
        self.current_seen_vars.add(node)
        for loop_node in self.current_loop:
            self.in_loop_vars[loop_node].add(node)
        self.generic_visit(node)

    def visit_For(self, node):
        self.current_loop.append(node)
        self.before_loop_vars[node] = copy.copy(self.current_seen_vars)
        self.generic_visit(node)
        self.current_loop.pop()

    def visit_While(self, node):
        self.current_loop.append(node)
        self.before_loop_vars[node] = copy.copy(self.current_seen_vars)
        self.generic_visit(node)
        self.current_loop.pop()


class LoopTransformer(gast.NodeTransformer):
    """
    This class transforms python while/for statement into Static Graph Ast
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of WhileTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.name_visitor = NameVisitor(self.root)

    def transform(self):
        self.visit(self.root)

    def get_for_stmt_nodes(self, node):
        self.generic_visit(node)
        # TODO
        return node

    def visit(self, node):
        self.generic_visit(node)
        # All parent nodes that may contain gast.While/gast.For
        if hasattr(node, 'body'):
            self.replace_stmt_list(node.body)
        if hasattr(node, 'orelse'):
            self.replace_stmt_list(node.orelse)
        return node

    def replace_stmt_list(self, body_list):
        if not isinstance(body_list, list):
            return

        i = 0
        while i < len(body_list):
            if isinstance(body_list[i], gast.While):
                new_stmts = self.get_while_stmt_nodes(body_list[i])
                body_list[i:i + 1] = new_stmts
                i += len(new_stmts)
            elif isinstance(body_list[i], gast.For):
                # TODO
                i += 1
            else:
                i += 1

    def get_while_stmt_nodes(self, node):
        # TODO: consider while - else in python
        # self.generic_visit(node)

        if not self.name_visitor.is_control_flow_loop(node):
            return [node]

        loop_var_names, create_var_names = self.name_visitor.get_loop_var_names(
            node)
        new_stmts = []

        # Python can create variable in loop and use it out of loop, E.g.
        #
        # while x < 10:
        #     x += 1
        #     y = x
        # z = y
        #
        # We need to create static variable for those variables
        for name in create_var_names:
            new_stmts.append(create_static_variable_gast_node(name))

        # while x < 10 in dygraph should be convert into static tensor < 10
        for name in loop_var_names:
            new_stmts.append(to_static_variable_gast_node(name))

        condition_func_node = gast.FunctionDef(
            name=unique_name.generate(WHILE_CONDITION_PREFIX),
            args=gast.arguments(
                args=[
                    gast.Name(
                        id=name,
                        ctx=gast.Param(),
                        annotation=None,
                        type_comment=None) for name in loop_var_names
                ],
                posonlyargs=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=None,
                kwarg=None,
                defaults=[]),
            body=[gast.Return(value=node.test)],
            decorator_list=[],
            returns=None,
            type_comment=None)
        new_stmts.append(condition_func_node)

        new_body = node.body
        new_body.append(
            gast.Return(value=generate_name_node(
                loop_var_names, ctx=gast.Load())))
        body_func_node = gast.FunctionDef(
            name=unique_name.generate(WHILE_BODY_PREFIX),
            args=gast.arguments(
                args=[
                    gast.Name(
                        id=name,
                        ctx=gast.Param(),
                        annotation=None,
                        type_comment=None) for name in loop_var_names
                ],
                posonlyargs=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=None,
                kwarg=None,
                defaults=[]),
            body=new_body,
            decorator_list=[],
            returns=None,
            type_comment=None)
        new_stmts.append(body_func_node)

        while_loop_node = create_while_node(condition_func_node.name,
                                            body_func_node.name, loop_var_names)
        new_stmts.append(while_loop_node)
        return new_stmts
