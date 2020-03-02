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
from .ast_utils import create_funcDef_node, generate_name_node
from .variable_trans_func import create_static_variable_gast_node, get_to_static_variable_gast_node

WHILE_CONDITION_PREFIX = 'while_condition'
WHILE_BODY_PREFIX = 'while_body'


def is_control_flow_loop(node):
    # TODO: make a better condition
    return True


def create_while_node(condition_name, body_name, loop_var_names):
    while_args = []
    while_args.append(gast.Name(id=condition_name, ctx=gast.Param()))
    while_args.append(gast.Name(id=body_name, ctx=gast.Param()))
    while_args.extend(
        gast.Name(
            id=var_name, ctx=gast.Param()) for var_name in loop_var_names)

    while_func_id = gast.parse('fluid.layers.while_loop').body[0].value
    while_node = gast.Call(func=while_func_id, args=while_args),
    assign_node = gast.Assign(
        generate_name_node(
            loop_var_names, ctx=gast.Store), while_node)
    return assign_node


class NameVistor(gast.NodeVisitor):
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

    def get_loop_var_names(self, node):
        assert isinstance(while_node, gast.While) or isinstance(
            while_node, gast.For), "Input node is not gast loop node"
        in_loop_vars = self.in_loop_vars[node]
        before_loop_vars = self.before_loop_vars[node]
        after_loop_var = self.current_seen_vars - in_loop_vars
        loop_var_names = set()
        create_var_names = set()
        read_context = {type(gast.Load), type(gast.AugLoad)}
        for name_node in in_loop_vars:
            if name_node in before_loop_vars:
                # If a variable is used in loop and created before loop, it
                # should be in loop_var as input
                loop_var_names.add(name_node.id)
            elif (name_node in after_loop_var) and (
                    type(name_node.ctx) in read_context):
                # If a variable is created in the while loop and read after
                # loop, it should be in loop_var and we should create it
                loop_var_names.add(name_node.id)
                create_var_names.add(name_node.id)
        return loop_var_names, create_var_names

    def visit_Name(self, node):
        self.current_seen_vars.add(node)
        for loop_node in self.current_loop:
            in_loop_vars[loop_node].add(node)
        self.generic_visit(node)

    def visit_For(self, node):
        self.current_loop.append(node)
        self.before_loop_vars[node] = copy.deepcopy(self.current_seen_vars)
        self.generic_visit(node)
        self.current_loop.pop()

    def visit_While(self, node):
        self.current_loop.append(node)
        self.before_loop_vars[node] = copy.deepcopy(self.current_seen_vars)
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
        self.name_visitor = NameVistor(self.root)

    def transform(self):
        self.visit(self.root)

    def visit_For(self, node):
        self.generic_visit(node)
        # TODO
        return node

    def visit_While(self, node):
        # TODO: consider while - else in python
        self.generic_visit(node)
        loop_var_names, create_var_names = self.name_visitor.create_var_names(
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
            args=gast.arguments(args=[
                gast.Name(
                    id=name, ctx=gast.Param()) for name in loop_var_names
            ]),
            body=gast.Return(value=node.test))
        new_stmts.append(condition_func_node)

        new_body = [
            node.body, generate_name_node(
                loop_var_names, ctx=gast.Load())
        ]
        body_func_node = gast.FunctionDef(
            name=unique_name.generate(WHILE_BODY_PREFIX),
            args=gast.arguments(args=[
                gast.Name(
                    id=name, ctx=gast.Param()) for name in loop_var_names
            ]),
            body=new_body)
        new_stmts.append(body_func_node)

        while_loop_node = create_while_node(condition_func_node.name,
                                            body_func_node.name, loop_var_names)
        new_stmts.append(while_loop_node)

        return gast.Suite(body=new_stmts)
