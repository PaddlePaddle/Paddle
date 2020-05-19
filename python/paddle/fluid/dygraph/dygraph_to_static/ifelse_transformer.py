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
from collections import defaultdict

# gast is a generic AST to represent Python2 and Python3's Abstract Syntax Tree(AST).
# It provides a compatibility layer between the AST of various Python versions,
# as produced by ast.parse from the standard ast module.
# See details in https://github.com/serge-sans-paille/gast/
import gast
from paddle.fluid import unique_name

from paddle.fluid.dygraph.dygraph_to_static.utils import compare_with_none
from paddle.fluid.dygraph.dygraph_to_static.utils import is_candidate_node
from paddle.fluid.dygraph.dygraph_to_static.utils import is_paddle_api
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import create_funcDef_node
from paddle.fluid.dygraph.dygraph_to_static.utils import create_assign_node
from paddle.fluid.dygraph.dygraph_to_static.utils import IsControlFlowVisitor
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import NodeVarType

TRUE_FUNC_PREFIX = 'true_fn'
FALSE_FUNC_PREFIX = 'false_fn'
LOGIC_AND_PREFIX = 'logic_and'
LOGIC_OR_PREFIX = 'logic_or'
LOGIC_NOT_PREFIX = 'logic_not'
PLAIN_TENSOR_PREFIX = 'bool_tensor'


class IfElseTransformer(gast.NodeTransformer):
    """
    Transform if/else statement of Dygraph into Static Graph.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Type of input node should be AstNodeWrapper, but received %s ." % type(
            wrapper_root)
        self.root = wrapper_root.node
        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.new_func_nodes = {}

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)
        self.after_visit(self.root)

    def visit_If(self, node):
        if_condition_visitor = IfConditionVisitor(node.test,
                                                  self.static_analysis_visitor)
        need_transform = if_condition_visitor.is_control_flow()
        self.generic_visit(node)
        if need_transform:
            pred_node, new_assign_nodes = if_condition_visitor.transform()
            true_func_node, false_func_node, return_name_ids = transform_if_else(
                node, self.root)
            # create layers.cond
            new_node = create_cond_node(return_name_ids, pred_node,
                                        true_func_node, false_func_node)
            self.new_func_nodes[new_node] = [true_func_node, false_func_node
                                             ] + new_assign_nodes
            return new_node
        else:
            return node

    def visit_Call(self, node):
        # Remove `numpy()` statement, like `Tensor.numpy()[i]` -> `Tensor[i]`
        if isinstance(node.func, gast.Attribute):
            attribute = node.func
            if attribute.attr == 'numpy':
                node = attribute.value
        self.generic_visit(node)
        return node

    def visit_IfExp(self, node):
        """
        Transformation with `true_fn(x) if Tensor > 0 else false_fn(x)`
        """
        if_condition_visitor = IfConditionVisitor(node.test,
                                                  self.static_analysis_visitor)
        need_transform = if_condition_visitor.is_control_flow()
        self.generic_visit(node)
        if need_transform:
            pred_node, new_assign_nodes = if_condition_visitor.transform()

            if len(new_assign_nodes) > 0:
                pred_node = merge_multi_assign_nodes(new_assign_nodes)

            new_node = create_cond_node(None, pred_node, node.body, node.orelse,
                                        True)
            # Note: A blank line will be added separately if transform gast.Expr
            # into source code. Using gast.Expr.value instead to avoid syntax error
            # in python.
            if isinstance(new_node, gast.Expr):
                new_node = new_node.value

            return new_node
        else:
            return node

    def after_visit(self, node):
        """
        This function will add some postprocessing operations with node.
        It can be used to add the created `true_fn/false_fn` in front of
        the node.body before they are called in cond layer.
        """
        self._insert_func_nodes(node)

    def _insert_func_nodes(self, node):
        """
        Defined `true_func` and `false_func` will be inserted in front of corresponding
        `layers.cond` statement instead of inserting them all into body of parent node.
        Because private variables of class or other external scope will be modified.
        For example, `self.var_dict["key"]`. In this case, nested structure of newly
        defined functions is easier to understand.
        """
        if not self.new_func_nodes:
            return
        idx = -1
        if isinstance(node, list):
            idx = len(node) - 1
        elif isinstance(node, gast.AST):
            for _, child in gast.iter_fields(node):
                self._insert_func_nodes(child)
        while idx >= 0:
            child_node = node[idx]
            if child_node in self.new_func_nodes:
                node[idx:idx] = self.new_func_nodes[child_node]
                idx = idx + len(self.new_func_nodes[child_node]) - 1
                del self.new_func_nodes[child_node]
            else:
                self._insert_func_nodes(child_node)
                idx = idx - 1

    def get_new_func_nodes(self):
        return self.new_func_nodes


def merge_multi_assign_nodes(assign_nodes):
    """
     Merges multiple separate assign statements into a single node.
    """
    if not isinstance(assign_nodes, (list, tuple)):
        assign_nodes = [assign_nodes]

    return MergeAssignTransformer().transform(assign_nodes)


class MergeAssignTransformer(gast.NodeTransformer):
    """
    Merges multiple separate assign statements into a single node.
    Because it cannot be determined the insertion location of new nodes for `IfExpr`,
    so replaces original node with merges conditional node.

    Note: This is a very low level api and only used for IfExpr transformation
          in control flow.

    For example:
        IfExpr:
            y = x+1 if mean or x > 0 else x-1

        assign nodes:
            bool_tensor_1 = fluid.layers.cast(x=mean, dtype='bool')
            logic_or_0 = fluid.layers.logical_or(x=bool_tensor_1, y=x > 0)

        merged node:
            fluid.layers.logical_or(x=fluid.layers.cast(x=mean, dtype='bool'), y=x > 0)
    """

    def __init__(self):
        self._name_to_nodes_value = {}

    def transform(self, nodes):
        value = None
        for node in nodes:
            assert isinstance(node, gast.Assign)
            # Note: targets of created assign node in control flow `if`
            # only contains one element.
            assert isinstance(node.targets[0], gast.Name)
            target_name = node.targets[0].id
            value = self.visit(node.value)
            self._name_to_nodes_value[target_name] = value

        return value

    def visit_Name(self, node):
        if node.id in self._name_to_nodes_value:
            node = self._name_to_nodes_value[node.id]
        return node


class NodeTestTransformer(gast.NodeTransformer):
    def __init__(self,
                 ast_node,
                 compare_nodes_with_tensor=None,
                 node_to_wrapper_map=None):
        if compare_nodes_with_tensor is None:
            compare_nodes_with_tensor = set()
        self.ast_root = ast_node
        self._compare_nodes_with_tensor = compare_nodes_with_tensor
        if node_to_wrapper_map is None:
            node_to_wrapper_map = {}
        self.node_to_wrapper_map = node_to_wrapper_map
        self._new_assign_nodes = []

    def transform(self):
        node = self.ast_root
        if not is_candidate_node(node):
            return self._create_cast_node(node)
        return self.visit(node)

    def visit_Call(self, node):
        # Remove `numpy()` statement, like `Tensor.numpy()[i]` -> `Tensor[i]`
        if isinstance(node.func, gast.Attribute):
            attribute = node.func
            if attribute.attr == 'numpy':
                node = attribute.value
        self.generic_visit(node)
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, gast.Not):
            arg = ast_to_source_code(node.operand)
            new_node_str = "fluid.layers.logical_not({})".format(arg)
            # gast.parse returns Module(body=[expr(value=...)])
            new_node = gast.parse(new_node_str).body[0].value
            logic_tensor_name = unique_name.generate(LOGIC_NOT_PREFIX)
            assign_name, assign_node = create_assign_node(logic_tensor_name,
                                                          new_node)
            self._new_assign_nodes.append(assign_node)
            return assign_name

        return node

    def visit_BoolOp(self, node):
        for i, child in enumerate(node.values):
            if not is_candidate_node(child):
                node_wrapper = self.node_to_wrapper_map.get(child, None)
                if node_wrapper and node_wrapper.node_var_type & NodeVarType.TENSOR_TYPES:
                    node.values[i] = self._create_cast_node(child)
                else:
                    node.values[i] = self._create_bool_node(child)
        self.generic_visit(node)
        new_node = self._create_logic_node(node)
        return new_node

    def visit_Compare(self, node):
        if compare_with_none(
                node) or node not in self._compare_nodes_with_tensor:
            return self._create_bool_node(node)
        self.generic_visit(node)
        return node

    def _create_cast_node(self, node):
        template = "fluid.layers.cast(x={}, dtype='bool')"

        return self._create_node_with_api_template(node, template)

    def _create_bool_node(self, node):
        template = "fluid.layers.fill_constant(shape=[1], dtype='bool', value=bool({}))"

        return self._create_node_with_api_template(node, template)

    def _create_node_with_api_template(self, node, template):
        node_code = ast_to_source_code(node)
        new_node_str = template.format(node_code)
        # gast.parse return Module(body=[expr(value=...)])
        new_node = gast.parse(new_node_str).body[0].value
        bool_tensor_name = unique_name.generate(PLAIN_TENSOR_PREFIX)
        assign_name, assign_node = create_assign_node(bool_tensor_name,
                                                      new_node)

        self._new_assign_nodes.append(assign_node)

        return assign_name

    def _create_logic_node(self, node):
        def _create_node(nodes, api_type):
            assert len(
                nodes
            ) > 1, "The length of BoolOp should be at least 2, but received {}.".format(
                len(nodes))
            if len(nodes) > 2:
                # Creates logic_and/logic_or node recursively.
                pre_assign_node = _create_node(nodes[:2], api_type)
                nodes = [pre_assign_node] + nodes[2:]
            args = [ast_to_source_code(child) for child in nodes]
            new_node_str = "fluid.layers.logical_{}(x={}, y={})".format(
                api_type, args[0], args[1])
            # gast.parse return Module(body=[expr(value=...)])
            new_node = gast.parse(new_node_str).body[0].value
            logic_tensor_name = unique_name.generate(
                LOGIC_AND_PREFIX if 'and' in api_type else LOGIC_OR_PREFIX)
            assign_name, assign_node = create_assign_node(logic_tensor_name,
                                                          new_node)
            self._new_assign_nodes.append(assign_node)

            return assign_name

        if isinstance(node.op, gast.And):
            node = _create_node(node.values, 'and')
        elif isinstance(node.op, gast.Or):
            node = _create_node(node.values, 'or')
        else:
            raise TypeError(
                "Only supports and/or syntax in control flow if statement.")
        return node

    def get_new_assign_nodes(self):
        return self._new_assign_nodes

    def set_compare_nodes_with_tensor(self, nodes_set):
        self._compare_nodes_with_tensor = set(nodes_set)
        return self._compare_nodes_with_tensor


class IfConditionVisitor(object):
    def __init__(self,
                 node,
                 static_analysis_visitor=None,
                 node_var_type_map=None):
        self.node = node
        self.static_analysis_visitor = static_analysis_visitor
        self.visitor = IsControlFlowVisitor(node, static_analysis_visitor,
                                            node_var_type_map)
        self.transformer = NodeTestTransformer(
            node, node_to_wrapper_map=self.visitor.node_to_wrapper_map)
        self.compare_nodes_with_tensor = set()
        self._is_control_flow_if = False

    def is_control_flow(self):
        """
        Determine whether the node is a plain python `if statement` or
        control flow in Paddle.
        """
        self._is_control_flow_if = self.visitor.transform()
        return self._is_control_flow_if

    def transform(self):
        if not self._is_control_flow_if:
            return self.node, []
        else:
            self.compare_nodes_with_tensor = self.visitor.get_compare_nodes_with_tensor(
            )
            self.transformer.set_compare_nodes_with_tensor(
                self.compare_nodes_with_tensor)
            new_node = self.transformer.transform()
            new_assign_nodes = self.transformer.get_new_assign_nodes()
            return new_node, new_assign_nodes


class NameVisitor(gast.NodeVisitor):
    def __init__(self, end_node=None):
        # The terminate node of the visitor.
        self.end_node = end_node
        # Dict to store the names and ctxs of vars.
        self.name_ids = defaultdict(list)
        # List of current visited nodes
        self.ancestor_nodes = []
        # Available only when end_node is set.
        self._is_finished = False
        self._candidate_ctxs = (gast.Store, gast.Load, gast.Param)

    def visit(self, node):
        """Visit a node."""
        if node == self.end_node or self._is_finished:
            self._is_finished = True
            return

        self.ancestor_nodes.append(node)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()

        return ret

    def visit_If(self, node):
        """
        For nested `if/else`, the created vars are not always visible for parent node.
        In addition, the vars created in `if.body` are not visible for `if.orelse`.

        Case 1:
            x = 1
            if m > 1:
                res = new_tensor
            res = res + 1   # Error, `res` is not visible here.

        Case 2:
            if x_tensor > 0:
                res = new_tensor
            else:
                res = res + 1   # Error, `res` is not visible here.

        In above two cases, we should consider to manage the scope of vars to parsing
        the arguments and returned vars correctly.
        """
        if not self.end_node:
            self.generic_visit(node)
        else:
            before_if_name_ids = copy.deepcopy(self.name_ids)
            body_name_ids = self._visit_child(node.body)
            # If traversal process stops early in `if.body`, return the currently seen name_ids.
            if self._is_finished:
                self._update_name_ids(before_if_name_ids)
            else:
                else_name_ids = self._visit_child(node.orelse)
                # If traversal process stops early in `if.orelse`, return the currently seen name_ids.
                if self._is_finished:
                    self._update_name_ids(before_if_name_ids)
                else:
                    # Blocks the vars in `if.body` and only inserts the vars both created in 'if/else' branch
                    # into name_ids.
                    new_name_ids = self._find_new_name_ids(body_name_ids,
                                                           else_name_ids)
                    for new_name_id in new_name_ids:
                        before_if_name_ids[new_name_id].append(gast.Store())

                    self.name_ids = before_if_name_ids

    def visit_Attribute(self, node):
        if not self._is_call_func_name_node(node):
            self.generic_visit(node)

    def visit_Name(self, node):
        blacklist = {'True', 'False', 'None'}
        if node.id in blacklist: return
        if not self._is_call_func_name_node(node):
            if isinstance(node.ctx, self._candidate_ctxs):
                self.name_ids[node.id].append(node.ctx)

    def visit_Assign(self, node):
        # Visit `value` firstly.
        node._fields = ('value', 'targets')
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not self.end_node:
            self.generic_visit(node)
        else:
            before_name_ids = copy.deepcopy(self.name_ids)
            self.name_ids = defaultdict(list)
            self.generic_visit(node)

            if self._is_finished:
                self._update_name_ids(before_name_ids)
            else:
                self.name_ids = before_name_ids

    def visit_Return(self, node):
        # Ignore the vars in return
        return

    def _visit_child(self, node):
        self.name_ids = defaultdict(list)
        if isinstance(node, list):
            for item in node:
                if isinstance(item, gast.AST):
                    self.visit(item)
        elif isinstance(node, gast.AST):
            self.visit(node)

        return copy.deepcopy(self.name_ids)

    def _find_new_name_ids(self, body_name_ids, else_name_ids):
        def is_required_ctx(ctxs, required_ctx):
            for ctx in ctxs:
                if isinstance(ctx, required_ctx):
                    return True
            return False

        candidate_name_ids = set(body_name_ids.keys()) & set(else_name_ids.keys(
        ))
        store_ctx = gast.Store
        new_name_ids = set()
        for name_id in candidate_name_ids:
            if is_required_ctx(body_name_ids[name_id],
                               store_ctx) and is_required_ctx(
                                   else_name_ids[name_id], store_ctx):
                new_name_ids.add(name_id)

        return new_name_ids

    def _is_call_func_name_node(self, node):
        if len(self.ancestor_nodes) > 1:
            assert self.ancestor_nodes[-1] == node
            parent_node = self.ancestor_nodes[-2]
            if isinstance(parent_node, gast.Call) and parent_node.func == node:
                return True
        return False

    def _update_name_ids(self, new_name_ids):
        for name_id, ctxs in new_name_ids.items():
            self.name_ids[name_id] = ctxs + self.name_ids[name_id]


def get_name_ids(nodes, end_node=None):
    """
    Return all ast.Name.id of python variable in nodes.
    """
    name_visitor = NameVisitor(end_node)
    for node in nodes:
        name_visitor.visit(node)
    return name_visitor.name_ids


def parse_cond_args(var_ids_dict, return_ids=None, ctx=gast.Load):
    """
    Find out the ast.Name.id list of input by analyzing node's AST information.
    """

    name_ids = [
        var_id for var_id, var_ctx in var_ids_dict.items()
        if isinstance(var_ctx[0], ctx)
    ]
    if return_ids:
        new_args = set(return_ids) - set(name_ids)
        name_ids.extend(list(new_args))
    name_ids.sort()
    args = [
        gast.Name(
            id=name_id, ctx=gast.Load(), annotation=None, type_comment=None)
        for name_id in name_ids
    ]
    arguments = gast.arguments(
        args=args,
        posonlyargs=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=None,
        kwarg=None,
        defaults=[])
    return arguments


def parse_cond_return(parent_vars_dict, if_vars_dict, else_vars_dict):
    """
    Find out the ast.Name list of output by analyzing node's AST information.
    Following conditions should be satisfied while determining whether a variable is a return value:
    1. the var in parent scope is modified in if/else node.
    2. new var is both created in if and else node.

    If different var is modified in if and else node, it should add the var in return_ids
    of different node.
    For example:
            x, y = 5, 10
            if x > 4:
                x = x+1
                z = x*x
            else:
                y = y - 1
                z = y*y

    The return_ids should be (x, y, z) for `if` and `else`node.
    """

    def _is_return_var(ctxs):
        for ctx in ctxs:
            if isinstance(ctx, (gast.Store, gast.Param)):
                return True
        return False

    def _vars_with_store(ids_dict):
        vars = []
        for k, ctxs in ids_dict.items():
            if _is_return_var(ctxs):
                vars.append(k)
        return vars

    def _candidate_vars(child_dict, parent_dict):
        return set([
            var for var in _vars_with_store(child_dict) if var in parent_dict
        ])

    # 1. the var in parent_ids is modified in if/else node.
    if_candidate_vars = _candidate_vars(if_vars_dict, parent_vars_dict)
    else_candidate_vars = _candidate_vars(else_vars_dict, parent_vars_dict)

    # 2. new var is both created in if and else node.
    if_new_vars = set([
        var for var in _vars_with_store(if_vars_dict)
        if var not in parent_vars_dict
    ])
    else_new_vars = set([
        var for var in _vars_with_store(else_vars_dict)
        if var not in parent_vars_dict
    ])
    new_vars = if_new_vars & else_new_vars

    # generate return_ids of if/else node.
    modified_vars = if_candidate_vars | else_candidate_vars
    return_ids = list(modified_vars | new_vars)
    return_ids.sort()

    return return_ids, list(modified_vars - new_vars)


def transform_if_else(node, root):
    """
    Transform ast.If into control flow statement of Paddle static graph.
    """
    parent_name_ids = get_name_ids([root], end_node=node)
    if_name_ids = get_name_ids(node.body)
    else_name_ids = get_name_ids(node.orelse)

    return_name_ids, modified_name_ids = parse_cond_return(
        parent_name_ids, if_name_ids, else_name_ids)

    true_func_node = create_funcDef_node(
        node.body,
        name=unique_name.generate(TRUE_FUNC_PREFIX),
        input_args=parse_cond_args(if_name_ids, modified_name_ids),
        return_name_ids=return_name_ids)
    false_func_node = create_funcDef_node(
        node.orelse,
        name=unique_name.generate(FALSE_FUNC_PREFIX),
        input_args=parse_cond_args(else_name_ids, modified_name_ids),
        return_name_ids=return_name_ids)

    return true_func_node, false_func_node, return_name_ids


def create_cond_node(return_name_ids,
                     pred,
                     true_func,
                     false_func,
                     is_if_expr=False):
    """
    Create `fluid.layers.cond(pred, true_fn, false_fn)` to replace
    original `python if/else` statement.
    """

    def create_lambda_node(func_or_expr_node, is_if_expr=False):
        body = func_or_expr_node
        if not is_if_expr:
            body = gast.Call(
                func=gast.Name(
                    id=func_or_expr_node.name,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None),
                args=[func_or_expr_node.args],
                keywords=[])

        lambda_node = gast.Lambda(
            args=gast.arguments(
                args=[],
                posonlyargs=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=None,
                kwarg=None,
                defaults=[]),
            body=body)
        return lambda_node

    cond_api = gast.parse('fluid.layers.cond').body[0].value
    true_func_lambda = create_lambda_node(true_func, is_if_expr)
    false_func_lambda = create_lambda_node(false_func, is_if_expr)
    cond_layer = gast.Call(
        func=cond_api,
        args=[pred, true_func_lambda, false_func_lambda],
        keywords=[])
    if return_name_ids:
        _, cond_node = create_assign_node(return_name_ids, cond_layer)
    else:  # No variables can be returned if no assign statement in if.body.
        cond_node = gast.Expr(value=cond_layer)

    return cond_node
