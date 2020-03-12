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

from paddle.fluid.dygraph.dygraph_to_static.utils import is_paddle_api
from paddle.fluid.dygraph.dygraph_to_static.utils import create_funcDef_node
from paddle.fluid.dygraph.dygraph_to_static.utils import generate_name_node
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper, NodeVarType

TRUE_FUNC_PREFIX = 'true_fn'
FALSE_FUNC_PREFIX = 'false_fn'


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
        assert isinstance(node, gast.If)
        need_transform = is_control_flow_if(node.test,
                                            self.static_analysis_visitor)
        self.generic_visit(node)
        if need_transform:
            pred_node = node.test
            true_func_node, false_func_node, return_name_ids = transform_if_else(
                node, self.root)
            # create layers.cond
            new_node = create_cond_node(return_name_ids, pred_node,
                                        true_func_node, false_func_node)
            self.new_func_nodes[new_node] = [true_func_node, false_func_node]
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
        self._insert_func_nodes(node)

    def _insert_func_nodes(self, parent_node):
        """
        Defined `true_func` and `false_func` will be inserted in front of corresponding
        `layers.cond` statement instead of inserting them all into body of parent node.
        Because private variables of class or other external scope will be modified.
        For example, `self.var_dict["key"]`. In this case, nested structure of newly
        defined functions is easier to understand.
        """
        if not (self.new_func_nodes and hasattr(parent_node, 'body')):
            return
        idx = len(parent_node.body) - 1
        while idx >= 0:
            child_node = parent_node.body[idx]
            if child_node in self.new_func_nodes:
                parent_node.body[idx:idx] = self.new_func_nodes[child_node]
                idx = idx + len(self.new_func_nodes[child_node]) - 1
                del self.new_func_nodes[child_node]
            else:
                self._insert_func_nodes(child_node)
                idx = idx - 1

    def get_new_func_nodes(self):
        return self.new_func_nodes


class IsControlFlowIfVisitor(gast.NodeTransformer):
    """
    Judge whether the node.test from Dygraph code dependent on paddle Tensor.
    If does, it should satisfy:
        1. must involve at least one var whose type is Tensor.
        2. the Tensor var should call `.numpy()[]` interface or Tensor.shape is [1].
        3. involve Tensor.shape[i] and the shape[i] is unknown in compile time.
    The following examples should not be considered as control_flow_if:
        1. `if Tensor_var` or `if Tensor_var is None`
        2. if Tensor.shape[i] is determined with fixed value (not -1 or None)

    Note: pred in ConditionalBlock require variable, which means all vars should be Tensor
          or transformed into Tensor, like fill_constant(shape=[1], dtype='int32', value=Tensor.shape[i]).

    TODO: 1. need to deal with `tensor.shape[i]` which need to eval the data of shape[i],
             because reshape_op may be called before this statement.
    """

    def __init__(self, static_analysis_visitor):
        self.static_analysis_visitor = static_analysis_visitor
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        self.is_control_flow = False

    def transform(self, node):
        if self._is_candidate_node(node):
            self.visit(node)
        return self.is_control_flow

    def visit_BoolOp(self, node):
        for child in node.values:
            if not self._is_candidate_node(child):
                continue
            self.generic_visit(node)
        return node

    def visit_Compare(self, node):
        # Ignores child node with `if x` or `if x is None`
        if not self._compare_with_none(node):
            self.generic_visit(node)
            for child in gast.walk(node):
                if isinstance(child, gast.Subscript):
                    self._visit_Subscript(child)
        return node

    def _visit_Subscript(self, node):
        self.generic_visit(node)
        if hasattr(node, 'value') and isinstance(node.value, gast.Call):
            self._visit_Call(node.value)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if isinstance(node.func, gast.Attribute):
            attr_node = node.func
            if attr_node.attr == 'numpy':
                self.is_control_flow = True

    def visit_Call(self, node):
        if is_paddle_api(node):
            self.is_control_flow = True
        return node

    def visit_Name(self, node):
        wrapper_node = self.node_to_wrapper_map.get(node, None)
        if wrapper_node is not None:
            if wrapper_node.node_var_type & {
                    NodeVarType.TENSOR, NodeVarType.PADDLE_RETURN_TYPES
            }:
                self.is_control_flow = True
        return node

    def _is_candidate_node(self, node):
        return isinstance(node, (gast.Compare, gast.BoolOp))

    def _compare_with_none(self, node):
        if isinstance(node, gast.Compare):
            for child in [node.left, node.comparators]:
                # node.comparators is a list.
                if isinstance(child, list):
                    child = child[0]
                if (isinstance(child, gast.Constant) and
                        child.value is None) or (
                            isinstance(child, gast.Name) and
                            child.id == 'None'):
                    return True
        return False


def is_control_flow_if(node, static_analysis_visitor=None):
    """
    Determine whether the node is a plain python `if statement` or
    control flow in Paddle.
    """
    assert isinstance(
        node, gast.AST
    ), "Type of input node should be gast.AST, but received %s." % type(node)
    if static_analysis_visitor is None:
        static_analysis_visitor = StaticAnalysisVisitor(node)
    return IsControlFlowIfVisitor(static_analysis_visitor).transform(node)


def get_name_ids(nodes, not_name_set=None, node_black_list=None):
    """
    Return all ast.Name.id of python variable in nodes.
    """
    if not isinstance(nodes, (list, tuple, set)):
        raise ValueError(
            "nodes must be one of list, tuple, set, but received %s" %
            type(nodes))
    if not_name_set is None:
        not_name_set = set()

    def update(old_dict, new_dict):
        for k, v in new_dict.items():
            old_dict[k].extend(v)

    name_ids = defaultdict(list)
    for node in nodes:
        if node_black_list and node in node_black_list:
            break
        if isinstance(node, gast.AST):
            # In two case, the ast.Name should be filtered.
            # 1. Function name like `my_func` of my_func(x)
            # 2. api prefix like `fluid` of `fluid.layers.mean`
            if isinstance(node, gast.Return):
                continue
            elif isinstance(node, gast.Call) and isinstance(node.func,
                                                            gast.Name):
                not_name_set.add(node.func.id)
            elif isinstance(node, gast.Attribute) and isinstance(node.value,
                                                                 gast.Name):
                not_name_set.add(node.value.id)
            if isinstance(
                    node, gast.Name
            ) and node.id not in name_ids and node.id not in not_name_set:
                if isinstance(node.ctx, (gast.Store, gast.Load, gast.Param)):
                    name_ids[node.id].append(node.ctx)
            else:
                if isinstance(node, gast.Assign):
                    node = copy.copy(node)
                    node._fields = ('value', 'targets')
                for field, value in gast.iter_fields(node):
                    value = value if isinstance(value, list) else [value]
                    update(name_ids,
                           get_name_ids(value, not_name_set, node_black_list))
    return name_ids


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
    parent_name_ids = get_name_ids([root], node_black_list=[node])
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


def create_cond_node(return_name_ids, pred, true_func, false_func):
    """
    Create `fluid.layers.cond(pred, true_fn, false_fn)` to replace
    original `python if/else` statement.
    """
    # TODO(Aurelius84): should replace the api hard code.
    cond_api = gast.parse('fluid.layers.cond').body[0].value
    true_func_lambda = gast.Lambda(
        args=gast.arguments(
            args=[],
            posonlyargs=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=None,
            kwarg=None,
            defaults=[]),
        body=gast.Call(
            func=gast.Name(
                id=true_func.name,
                ctx=gast.Load(),
                annotation=None,
                type_comment=None),
            args=[true_func.args],
            keywords=[]))
    false_func_lambda = gast.Lambda(
        args=gast.arguments(
            args=[],
            posonlyargs=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=None,
            kwarg=None,
            defaults=[]),
        body=gast.Call(
            func=gast.Name(
                id=false_func.name,
                ctx=gast.Load(),
                annotation=None,
                type_comment=None),
            args=[false_func.args],
            keywords=[]))
    cond_layer = gast.Call(
        func=cond_api,
        args=[pred, true_func_lambda, false_func_lambda],
        keywords=[])
    if return_name_ids:
        targets = [generate_name_node(return_name_ids, ctx=gast.Store())]
        assign_node = gast.Assign(targets=targets, value=cond_layer)
        return assign_node
    else:
        return gast.Expr(value=cond_layer)
