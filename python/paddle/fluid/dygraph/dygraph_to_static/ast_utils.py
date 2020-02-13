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

import ast
import six
import codegen
import copy
import tempfile
import imp
import os
import atexit
from collections import defaultdict

from paddle.fluid import unique_name

TRUE_FUNC_PRFIX = 'true_fn'
FALSE_FUNC_PRFIX = 'false_fn'


def is_control_flow_if(node):
    """
    Determine whether the node is a plain python `if statement` or
    control flow in Paddle.
    """
    return True


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
        # return old_dict

    name_ids = defaultdict(list)
    for node in nodes:
        if node_black_list and node in node_black_list: continue
        if isinstance(node, ast.AST):
            # In two case, the ast.Name should be filtered.
            # 1. Function name like `my_func` of my_func(x)
            # 2. api prefix like `fluid` of `fluid.layers.mean`
            if isinstance(node, ast.Return):
                continue
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                not_name_set.add(node.func.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value,
                                                                ast.Name):
                not_name_set.add(node.value.id)
            if isinstance(
                    node, ast.Name
            ) and node.id not in name_ids and node.id not in not_name_set:
                if isinstance(node.ctx, (ast.Store, ast.Load)):
                    name_ids[node.id].append(node.ctx)
            else:
                if isinstance(node, ast.Assign):
                    node = copy.copy(node)
                    node._fields = ('value', 'targets')
                for field, value in ast.iter_fields(node):
                    value = value if isinstance(value, list) else [value]
                    update(name_ids,
                           get_name_ids(value, not_name_set, node_black_list))
    return name_ids


def parse_args(var_ids_dict, return_ids=None, ctx=ast.Load):
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
    args = [ast.Name(id=name_id, ctx=ast.Load()) for name_id in name_ids]
    arguments = ast.arguments(args=args, vararg=None, kwarg=None, defaults=[])
    return arguments


def parse_return(parent_vars_dict, if_vars_dict, else_vars_dict):
    """
    Find out the ast.Name list of output by analyzing node's AST information.
    Following conditions should be satisfied while determining whether a variable is a return value:
    1. the var in parent_ids is modified in if/else node.
    2. new var is both created in if and else node.

    If different var is modified in if and else node, it should place `None` in return_ids
    of different node.
    For example:
            x, y = 5, 10
            if x > 4:
                x = x+1
                z = x*x
            else:
                y = y - 1
                z = y*y

    The return_ids should be (x, None, z) for `if` node and (None, y, z) for `else` node.
    """

    def _is_return_var(ctxs):
        for ctx in ctxs:
            if isinstance(ctx, ast.Store):
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


def create_funcDef_node(nodes, name, input_args, return_name_ids):
    """
    Wrapper all statements of nodes into one ast.FunctionDef, which can be
    called by ast.Call.
    """
    nodes = copy.copy(nodes)
    # add return statement
    nodes.append(ast.Return(value=generate_name_node(return_name_ids)))
    func_def_node = ast.FunctionDef(
        name=name, args=input_args, body=nodes, decorator_list=[])
    return func_def_node


def transform_if_else(node, root):
    """
    Transoform ast.If into control flow statement of Paddle static graph.
    """
    parent_name_ids = get_name_ids([root], node_black_list=[node])
    if_name_ids = get_name_ids(node.body)
    else_name_ids = get_name_ids(node.orelse)

    return_name_ids, modified_name_ids = parse_return(
        parent_name_ids, if_name_ids, else_name_ids)

    true_func_node = create_funcDef_node(
        node.body,
        name=unique_name.generate(TRUE_FUNC_PRFIX),
        input_args=parse_args(if_name_ids, modified_name_ids),
        return_name_ids=return_name_ids)
    false_func_node = create_funcDef_node(
        node.orelse,
        name=unique_name.generate(FALSE_FUNC_PRFIX),
        input_args=parse_args(else_name_ids, modified_name_ids),
        return_name_ids=return_name_ids)

    return true_func_node, false_func_node, return_name_ids


def create_cond_node(return_name_ids, pred, true_func, false_func):
    """
    Create `fluid.layers.cond(pred, true_fn, false_fn)` to replace
    original `python if/else` statement.
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
    targets = [generate_name_node(return_name_ids, ctx=ast.Store())]
    assign_node = ast.Assign(targets=targets, value=cond_layer)

    return assign_node


def ast_to_func(ast_root, func_name, delete_on_exit=True):
    """
    Transform modified AST of decorated function into python callable object.
    """
    source = codegen.to_source(ast_root)
    if six.PY2:
        source = source.encode('utf-8')
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    else:
        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8')

    # TODO: more elegent way to transform ast into callable object
    import_str = "import paddle\n" \
                 "import paddle.fluid as fluid\n" \
                 "import paddle.fluid.layers as layers\n"
    with f:
        module_name = os.path.basename(f.name[:-3])
        f.write(import_str)
        f.write(source)

    if delete_on_exit:
        atexit.register(lambda: os.remove(f.name))
    module = imp.load_source(module_name, f.name)
    if not hasattr(module, func_name):
        raise ValueError(
            'Function: %s doesn\'t exist in the Module transformed from AST.' %
            func_name)

    return getattr(module, func_name), f.name
