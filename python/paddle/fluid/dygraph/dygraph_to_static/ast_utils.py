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

import ast
import six
import codegen
import copy


def is_control_flow_if(node):
    """
    Determine whether the node is a plain python `if statement` or
    control flow in Paddle.
    """
    return True


def _all_name_ids(nodes, not_name_set=set()):
    if not isinstance(nodes, (list, tuple, set)):
        raise ValueError(
            "nodes must be one of list, tuple, set, but received %s" %
            type(nodes))

    name_ids = dict()
    for node in nodes:
        if isinstance(node, ast.AST):
            # In two case, the ast.Name should be filtered.
            # 1. Function name like `my_func` of my_func(x)
            # 2. api prefix like `fluid` of `fluid.layers.mean`
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                not_name_set.add(node.func.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value,
                                                                ast.Name):
                not_name_set.add(node.value.id)
            if isinstance(
                    node, ast.Name
            ) and node.id not in name_ids and node.id not in not_name_set:
                name_ids[node.id] = node.ctx
            else:
                if isinstance(node, ast.Assign):
                    node = copy.copy(node)
                    node._fields = ('value', 'targets')
                for field, value in ast.iter_fields(node):
                    value = value if isinstance(value, list) else [value]
                    name_ids = dict(
                        _all_name_ids(value, not_name_set).items() +
                        name_ids.items())
    return name_ids


def parse_args(var_ids_dict, ctx=ast.Load):
    """
    Find out the ast.Name.id list of input by analyzing node's AST information.
    """

    name_ids = [
        var_id for var_id, var_ctx in var_ids_dict.items()
        if isinstance(var_ctx, ctx)
    ]
    args = [ast.Name(id=name_id, ctx=ast.Load()) for name_id in name_ids]
    arguments = ast.arguments(args=args, vararg=None, kwarg=None, defaults=[])
    return arguments, name_ids


def parse_return(var_ids_dict):
    """
    Find out the ast.Name list of output by analyzing node's AST information.
    """
    name_ids = [
        var_id for var_id, ctx in var_ids_dict.items()
        if isinstance(ctx, ast.Store)
    ]
    return name_ids


def generate_name_node(name_ids, ctx=ast.Load()):
    if isinstance(name_ids, six.string_types):
        name_ids = [name_ids]
    if not isinstance(name_ids, (list, tuple, set)):
        raise TypeError('name_ids must be list or tuple or set, but received %s'
                        % type(type(name_ids)))
    ast_names = [ast.Name(id=name_id, ctx=ctx) for name_id in name_ids]
    if len(ast_names) == 1:
        name_node = ast_names[0]
    else:
        name_node = ast.Tuple(elts=ast_names)
    return name_node


def wrapper_to_func(nodes, name, return_name_ids=None):
    """
    Wrapper all statements of nodes into one ast.FunctionDef, which can be
    called by ast.Call.
    """
    nodes = copy.copy(nodes)
    all_var_names = _all_name_ids(nodes)
    input_args, _ = parse_args(all_var_names)
    if not nodes:
        if return_name_ids:
            return_name_ids = ['None'] * len(return_name_ids)
        else:
            raise ValueError(
                'nodes and return_name_ids shall not be both None or [], as least one should be specified.'
            )
    if not return_name_ids:
        return_name_ids = parse_return(all_var_names)
    # add return statement
    nodes.append(ast.Return(value=generate_name_node(return_name_ids)))
    func_def_node = ast.FunctionDef(
        name=name,
        args=input_args,
        body=nodes,
        decorator_list=[], )
    # print(codegen.to_source(func_def_node))
    return func_def_node, return_name_ids


def create_cond_node(targets, pred, true_func, false_func):
    """
    Create `fluid.layers.cond(pred, true_fn, false_fn)` to replace
    original `python if` statement.
    """
    #TODO: how to determine the statement of api (fluid.layers.cond or layers.cond or f.layers.cond)?
    cond_api = ast.parse('fluid.layers.cond').body[0].value
    true_func_lambda = ast.Lambda(
        args=ast.arguments(
            args=[], vararg=None, kwarg=None, defaults=[]),
        body=ast.Call(
            func=ast.Name(
                id=true_func.name, ctx=ast.Load()),
            args=[true_func.args],
            keywords=[],
            kwargs=None,
            starargs=None))
    false_func_lambda = ast.Lambda(
        args=ast.arguments(
            args=[], vararg=None, kwarg=None, defaults=[]),
        body=ast.Call(
            func=ast.Name(
                id=false_func.name, ctx=ast.Load()),
            args=[false_func.args],
            keywords=[],
            kwargs=None,
            starargs=None))
    cond_layer = ast.Call(
        func=cond_api,
        args=[pred, true_func_lambda, false_func_lambda],
        keywords=[],
        kwargs=None,
        starargs=None)
    targets = [generate_name_node(targets, ctx=ast.Store())]
    assign_node = ast.Assign(targets=targets, value=cond_layer)

    return assign_node


def is_numpy_slice(node):
    assert isinstance(node, ast.Subscript)
    if isinstance(node.value, ast.Call) and isinstance(node.value.func,
                                                       ast.Attribute):
        attribute = node.value.func
        return attribute.attr == 'numpy'
    return False


def wrapper_slice(node):
    """
    Transform `x[i]` into fluid.layer.slice(x, [i], [i])
    """
    if not getattr(node, 'slice', None):
        return node
    assert isinstance(node.slice.value, ast.Num)
    index = ast.List(elts=[node.slice.value], ctx=ast.Load())
    kargs = [
        ast.keyword(
            arg='starts', value=index), ast.keyword(
                arg='ends', value=index)
    ]
    slice_api = ast.parse('fluid.layers.slice').body[0].value
    new_call = ast.Call(
        func=slice_api,
        args=[node.value],
        keywords=kargs,
        starargs=ast.Name(
            id='args', ctx=ast.Param()),
        kwargs=ast.Name(
            id='kwargs', ctx=ast.Param()))
    return new_call
