# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import copy

from paddle.jit.dy2static.ast_utils import ast_to_source_code
from paddle.utils import gast


class BaseNodeVisitor(gast.NodeVisitor):
    """
    Implement customized NodeVisitor inherited from gast.NodeVisitor.
    Ancestor nodes are traced to easily support more operations of currently
    visited node.
    """

    def __init__(self):
        self.ancestor_nodes = []

    def visit(self, node):
        """Visit a node."""
        self.ancestor_nodes.append(node)

        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()
        return ret


def create_undefined_var(name):
    func_code = f"{name} = _jst.UndefinedVar('{name}')"
    return gast.parse(func_code).body[0]


def create_bool_node(name, value):
    '''
    Create a assign stmt for name = value .
    '''
    assert isinstance(value, bool)
    node = f"{name} = {value}"
    return gast.parse(node).body[0]


def get_parent_mapping(root):
    to_parent: dict[gast.AST, gast.AST] = {}
    for node in gast.walk(root):
        for child in gast.iter_child_nodes(node):
            to_parent[child] = node
    return to_parent


def create_name_str(name_ids):
    """
    Return "('x', 'y')" for [x, y]
    """
    if not name_ids:
        return 'None'

    names_str = ["'%s'" % (name.replace("'", "\\'")) for name in name_ids]
    return "(%s, )" % ','.join(names_str)


def create_function_def_node(nodes, name, input_args, return_name_ids):
    """
    Wrapper all statements of nodes into one ast.FunctionDef, which can be
    called by ast.Call.
    """
    nodes = copy.copy(nodes)
    # add return statement
    if return_name_ids:
        nodes.append(gast.Return(value=generate_name_node(return_name_ids)))
    else:
        nodes.append(gast.Return(value=None))
    func_def_node = gast.FunctionDef(
        name=name,
        args=input_args,
        body=nodes,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    return func_def_node


def create_assign_node(name, node):
    """
    Creates a `gast.Assign` node by given name_id as target and node as value.
    """
    targets = generate_name_node(name, ctx=gast.Store())
    assign_node = gast.Assign(targets=[targets], value=node)
    return targets, assign_node


def generate_name_node(name_ids, ctx=gast.Load(), gen_tuple_if_single=False):
    """
    If name_ids is list or tuple or set with multiple strings, this function
    generates gast.Tuple of gast.Name.
    If the name_ids is single string or contains only 1 string, this function
    returns gast.Name if gen_tuple_if_single==False else returns gast.Tuple
    with only one gast.Name

    This function is used at several gast.Return statements.
    """
    if isinstance(name_ids, str):
        name_ids = [name_ids]
    if not isinstance(name_ids, (list, tuple, set)):
        raise TypeError(
            'name_ids must be list or tuple or set, but received %s'
            % type(type(name_ids))
        )

    def create_node_for_name(name):
        if '.' not in name:
            return gast.Name(
                id=name, ctx=ctx, annotation=None, type_comment=None
            )
        return gast.parse(name).body[0].value

    gast_names = [create_node_for_name(name_id) for name_id in name_ids]
    if len(gast_names) == 1 and not gen_tuple_if_single:
        name_node = gast_names[0]
    else:
        name_node = gast.Tuple(elts=gast_names, ctx=ctx)
    return name_node


def get_attribute_full_name(node):
    assert isinstance(
        node, gast.Attribute
    ), "Input non-Attribute node to get attribute full name"
    return ast_to_source_code(node).strip()
