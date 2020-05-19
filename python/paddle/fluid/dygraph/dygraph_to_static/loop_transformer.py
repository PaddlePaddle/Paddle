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
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import NodeVarType
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import create_api_shape_node
from paddle.fluid.dygraph.dygraph_to_static.utils import generate_name_node
from paddle.fluid.dygraph.dygraph_to_static.utils import get_constant_variable_node
from paddle.fluid.dygraph.dygraph_to_static.utils import get_attribute_full_name
from paddle.fluid.dygraph.dygraph_to_static.utils import is_control_flow_to_transform
from paddle.fluid.dygraph.dygraph_to_static.utils import ForNodeVisitor
from paddle.fluid.dygraph.dygraph_to_static.utils import RenameTransformer
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import create_static_variable_gast_node
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import to_static_variable_gast_node

__all__ = ['LoopTransformer', 'NameVisitor']

WHILE_CONDITION_PREFIX = 'while_condition'
WHILE_BODY_PREFIX = 'while_body'

FOR_CONDITION_PREFIX = 'for_loop_condition'
FOR_BODY_PREFIX = 'for_loop_body'
GENERATE_VARIABLE_PREFIX = 'generate_variable'


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


class LogicalOpTransformer(gast.NodeTransformer):
    """
    Transform python boolean op into Paddle logical op
    """

    def __init__(self, node):
        self.root = node

    def transform(self):
        return self.visit(self.root)

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, gast.Not):
            arg = ast_to_source_code(node.operand)
            new_node_str = "fluid.layers.logical_not({})".format(arg)
            # gast.parse returns Module(body=[expr(value=...)])
            new_node = gast.parse(new_node_str).body[0].value
            return new_node
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, gast.And):
            new_node = self._create_bool_op_node(node.values, 'and')
        elif isinstance(node.op, gast.Or):
            new_node = self._create_bool_op_node(node.values, 'or')
        else:
            raise TypeError(
                "Only supports and/or syntax in control flow if statement.")
        return new_node

    def _create_bool_op_node(self, nodes, api_type):
        assert len(
            nodes
        ) > 1, "The length of BoolOp should be at least 2, but received {}.".format(
            len(nodes))
        if len(nodes) > 2:
            # Creates logic_and/logic_or node recursively.
            pre_assign_node = self._create_bool_op_node(nodes[:2], api_type)
            nodes = [pre_assign_node] + nodes[2:]
        args = [ast_to_source_code(child) for child in nodes]
        new_node_str = "fluid.layers.logical_{}(x={}, y={})".format(
            api_type, args[0], args[1])
        # gast.parse return Module(body=[expr(...)])
        new_node = gast.parse(new_node_str).body[0].value
        return new_node


class NameVisitor(gast.NodeVisitor):
    '''
    Analysis name liveness for loop transformer
    '''

    def __init__(self, root_node):
        # Set of gast.Name or gast.Attribute for variables
        self.current_seen_vars = set()

        # List of gast.While/gast.For nodes
        self.current_loop = []

        # List of nodes that have scope of variables.
        self.nodes_with_scope = []

        self.blacklist_names = {"False", "True", "None"}

        # Mapping from gast.While/gast.For to variable nodes
        self.before_loop_body_vars = defaultdict(set)
        self.in_loop_vars = defaultdict(set)

        # Mapping from gast.While/gast.For to variable nodes which is condition
        # of loop or being modified during the loop
        self.write_in_loop = defaultdict(set)
        self.condition_vars = defaultdict(set)
        self.in_condition = False

        self.static_analysis_visitor = StaticAnalysisVisitor(root_node)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )

        self.visit(root_node)

    def is_control_flow_loop(self, node):
        need_transform = is_control_flow_to_transform(
            node, self.static_analysis_visitor)
        return need_transform

    def get_loop_var_names(self, node):
        assert isinstance(
            node, (gast.While, gast.For)), "Input node is not gast loop node"
        loop_var_names = set()
        create_var_names = set()
        read_context = {type(gast.Load()), type(gast.AugLoad())}

        in_loop_vars = self.in_loop_vars[node]
        in_loop_name_strs = self._var_nodes_to_names(in_loop_vars)
        before_loop_body_vars = self.before_loop_body_vars[node]
        before_loop_name_strs = self._var_nodes_to_names(before_loop_body_vars)
        after_loop_vars = self.current_seen_vars - before_loop_body_vars - in_loop_vars
        after_loop_name_strs = self._var_nodes_to_names(after_loop_vars,
                                                        read_context)
        condition_vars = self.condition_vars[node]
        condition_names = self._var_nodes_to_names(condition_vars)
        write_vars = self.write_in_loop[node]
        write_names = self._var_nodes_to_names(write_vars)

        name_to_type = {}
        for var in in_loop_vars:
            wrapper = self.node_to_wrapper_map[var]
            name_to_type[self._var_node_to_name(var)] = wrapper.node_var_type

        for name in in_loop_name_strs:
            if name in before_loop_name_strs:
                # If a variable is used in loop and created before loop

                # If this var is a basic variable and read-only and not
                # condition var, it may not be loop_var else it should
                # be in loop_var as input
                if (not name in condition_names) and (
                        not name in write_names
                ) and self._node_var_type_is_basic(name_to_type[name]):
                    continue
                loop_var_names.add(name)

            elif name in after_loop_name_strs:
                # If a variable is created in the while loop and read after
                # loop, it should be in loop_var and we should create it

                # because name in after_loop_name must be initialized in loop
                # So it is write-only, we don't have to filter read-only basic
                # vars out
                loop_var_names.add(name)
                create_var_names.add(name)
        return loop_var_names, create_var_names

    def visit_Name(self, node):
        if self._is_call_func_name_node(node):
            self.generic_visit(node)
            return
        if node.id in self.blacklist_names:
            self.generic_visit(node)
            return

        self.current_seen_vars.add(node)
        write_context = {
            type(gast.Store()), type(gast.AugStore()), type(gast.Del())
        }
        for loop_node in self.current_loop:
            self.in_loop_vars[loop_node].add(node)
            if type(node.ctx) in write_context:
                self.write_in_loop[loop_node].add(node)
            if self.in_condition:
                self.condition_vars[loop_node].add(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.nodes_with_scope.append(node)
        self.blacklist_names.add(node.name)
        # The variables in the function are not visible to the outside scope.
        before_func_seen_vars = copy.copy(self.current_seen_vars)

        self.generic_visit(node)
        self.nodes_with_scope.pop()
        # After exiting the scope of the node, variables in this scope
        # should be removed from self.current_seen_vars.
        if self.nodes_with_scope:
            self.current_seen_vars = before_func_seen_vars

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        return ret

    def visit_Attribute(self, node):
        if self._is_call_func_name_node(node):
            return
        attr_full_name = get_attribute_full_name(node)
        # Class variables are not allowed to appear in the arguments list
        # of defined function under class methods in Python.
        """
        def class_func(self):
            def while_loop_body(self.x, y) # `self.x` is illegal.
        """
        # TODO: If do change the variable with `self.var`, need a better
        # way to deal with this case.
        if attr_full_name.startswith("self."):
            return
        self.current_seen_vars.add(node)

        for loop_node in self.current_loop:
            self.in_loop_vars[loop_node].add(node)

        # sub-nodes are visited during get_attribute_full_name and we shouldn't
        # visit again

    def visit_For(self, node):
        self.current_loop.append(node)
        self.in_condition = True
        self.visit(node.target)
        self.visit(node.iter)
        self.in_condition = False
        self.before_loop_body_vars[node] = copy.copy(self.current_seen_vars)
        self.generic_visit(node)
        self.current_loop.pop()

    def visit_While(self, node):
        self.current_loop.append(node)
        self.in_condition = True
        self.visit(node.test)
        self.in_condition = False
        self.before_loop_body_vars[node] = copy.copy(self.current_seen_vars)
        self.generic_visit(node)
        self.current_loop.pop()

    def _var_nodes_to_names(self, node_set, ctx_filter_set=None):
        ret = set()
        for node in node_set:
            if ctx_filter_set is None or type(node.ctx) in ctx_filter_set:
                ret.add(self._var_node_to_name(node))
        return ret

    def _var_node_to_name(self, node):
        if isinstance(node, gast.Name):
            return node.id
        elif isinstance(node, gast.Attribute):
            return get_attribute_full_name(node)

    def _node_var_type_is_basic(self, node_var_type):
        basic_types = {
            NodeVarType.BOOLEAN, NodeVarType.INT, NodeVarType.FLOAT,
            NodeVarType.STRING
        }
        for t in node_var_type:
            if t in basic_types:
                return True
        return False

    def _is_call_func_name_node(self, node):
        parent_node = self.node_to_wrapper_map[node].parent.node
        if isinstance(parent_node, gast.Call) and parent_node.func == node:
            return True
        return False


class LoopTransformer(gast.NodeTransformer):
    """
    This class transforms python while/for statement into Static Graph Ast
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of LoopTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.name_visitor = NameVisitor(self.root)

    def transform(self):
        self.visit(self.root)

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
                new_stmts = self.get_for_stmt_nodes(body_list[i])
                body_list[i:i + 1] = new_stmts
                i += len(new_stmts)
            else:
                i += 1

    def get_for_stmt_nodes(self, node):
        # TODO: consider for - else in python

        # 1. check whether need to transform
        # NOTE: Current need transform cases:
        #   1). for x in range(VarBase.numpy()[0])
        #   2). for x in VarBase.numpy()
        #   3). for i, x in enumerate(VarBase.numpy())
        if not self.name_visitor.is_control_flow_loop(node):
            return [node]

        # 2. get key statements for different cases
        # NOTE: three key statements:
        #   1). init_stmts: list[node], prepare nodes of for loop, may not only one
        #   2). cond_stmt: node, condition node to judge whether continue loop
        #   3). body_stmts: list[node], updated loop body, sometimes we should change
        #       the original statement in body, not just append new statement
        current_for_node_parser = ForNodeVisitor(node)
        stmts_tuple = current_for_node_parser.parse()
        if stmts_tuple is None:
            return [node]
        init_stmts, cond_stmt, body_stmts = stmts_tuple

        # 3. get original loop vars
        loop_var_names, create_var_names = self.name_visitor.get_loop_var_names(
            node)
        # NOTE: in 'for x in var' or 'for i, x in enumerate(var)' cases,
        # we need append new loop var & remove useless loop var
        #   1. for x in var -> x is no need
        #   2. for i, x in enumerate(var) -> x is no need
        if current_for_node_parser.is_for_iter(
        ) or current_for_node_parser.is_for_enumerate_iter():
            iter_var_name = current_for_node_parser.iter_var_name
            iter_idx_name = current_for_node_parser.iter_idx_name
            loop_var_names.add(iter_idx_name)
            if iter_var_name not in create_var_names:
                loop_var_names.remove(iter_var_name)

        # 4. prepare result statement list
        new_stmts = []
        # Python can create variable in loop and use it out of loop, E.g.
        #
        # for x in range(10):
        #     y += x
        # print(x) # x = 10
        #
        # We need to create static variable for those variables
        for name in create_var_names:
            if "." not in name:
                new_stmts.append(create_static_variable_gast_node(name))

        # 5. append init statements
        new_stmts.extend(init_stmts)
        # for x in range(10) in dygraph should be convert into static tensor + 1 <= 10
        for name in loop_var_names:
            new_stmts.append(to_static_variable_gast_node(name))

        # 6. create & append condition function node
        condition_func_node = gast.FunctionDef(
            name=unique_name.generate(FOR_CONDITION_PREFIX),
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
            body=[gast.Return(value=cond_stmt)],
            decorator_list=[],
            returns=None,
            type_comment=None)
        for name in loop_var_names:
            if "." in name:
                rename_transformer = RenameTransformer(condition_func_node)
                rename_transformer.rename(
                    name, unique_name.generate(GENERATE_VARIABLE_PREFIX))
        new_stmts.append(condition_func_node)

        # 7. create & append loop body function node
        # append return values for loop body
        body_stmts.append(
            gast.Return(value=generate_name_node(
                loop_var_names, ctx=gast.Load())))
        body_func_node = gast.FunctionDef(
            name=unique_name.generate(FOR_BODY_PREFIX),
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
            body=body_stmts,
            decorator_list=[],
            returns=None,
            type_comment=None)
        for name in loop_var_names:
            if "." in name:
                rename_transformer = RenameTransformer(body_func_node)
                rename_transformer.rename(
                    name, unique_name.generate(GENERATE_VARIABLE_PREFIX))
        new_stmts.append(body_func_node)

        # 8. create & append while loop node 
        while_loop_node = create_while_node(condition_func_node.name,
                                            body_func_node.name, loop_var_names)
        new_stmts.append(while_loop_node)

        return new_stmts

    def get_while_stmt_nodes(self, node):
        # TODO: consider while - else in python
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
            if "." not in name:
                new_stmts.append(create_static_variable_gast_node(name))

        # while x < 10 in dygraph should be convert into static tensor < 10
        for name in loop_var_names:
            new_stmts.append(to_static_variable_gast_node(name))

        logical_op_transformer = LogicalOpTransformer(node.test)
        cond_value_node = logical_op_transformer.transform()

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
            body=[gast.Return(value=cond_value_node)],
            decorator_list=[],
            returns=None,
            type_comment=None)
        for name in loop_var_names:
            if "." in name:
                rename_transformer = RenameTransformer(condition_func_node)
                rename_transformer.rename(
                    name, unique_name.generate(GENERATE_VARIABLE_PREFIX))
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
        for name in loop_var_names:
            if "." in name:
                rename_transformer = RenameTransformer(body_func_node)
                rename_transformer.rename(
                    name, unique_name.generate(GENERATE_VARIABLE_PREFIX))
        new_stmts.append(body_func_node)

        while_loop_node = create_while_node(condition_func_node.name,
                                            body_func_node.name, loop_var_names)
        new_stmts.append(while_loop_node)
        return new_stmts
