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
import ast
import gast
import six
import copy
import tempfile
import imp
import os
import atexit
from collections import defaultdict

from paddle.fluid import unique_name

TRUE_FUNC_PREFIX = 'true_fn'
FALSE_FUNC_PREFIX = 'false_fn'


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

    def __init__(self, node):
        self.node = node
        self.is_control_flow = False

    def ast_visit(self):
        self.visit(self.node)
        return self.is_control_flow

    def visit_Compare(self, node):
        for child in gast.walk(node):
            if isinstance(child, gast.Subscript):
                self._visit_Subscript(child)
        return node

    def _visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, gast.Call):
            self._visit_Call(node.value)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if isinstance(node.func, gast.Attribute):
            attr_node = node.func
            self.is_control_flow = (attr_node.attr == 'numpy')


def is_control_flow_if(node):
    """
    Determine whether the node is a plain python `if statement` or
    control flow in Paddle.
    """
    assert isinstance(
        node, gast.AST
    ), "Type of input node should be gast.AST, but received %s." % type(node)
    return IsControlFlowIfVisitor(node).ast_visit()


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
        if node_black_list and node in node_black_list: continue
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


def generate_name_node(name_ids, ctx=gast.Load()):
    """
    Generate list or gast.Tuple of ast.Name for Return statement.
    """
    if isinstance(name_ids, six.string_types):
        name_ids = [name_ids]
    if not isinstance(name_ids, (list, tuple, set)):
        raise TypeError('name_ids must be list or tuple or set, but received %s'
                        % type(type(name_ids)))
    gast_names = [
        gast.Name(
            id=name_id, ctx=ctx, annotation=None, type_comment=None)
        for name_id in name_ids
    ]
    if len(gast_names) == 1:
        name_node = gast_names[0]
    else:
        name_node = gast.Tuple(elts=gast_names, ctx=ctx)
    return name_node


def create_funcDef_node(nodes, name, input_args, return_name_ids):
    """
    Wrapper all statements of nodes into one ast.FunctionDef, which can be
    called by ast.Call.
    """
    nodes = copy.copy(nodes)
    # add return statement
    nodes.append(gast.Return(value=generate_name_node(return_name_ids)))
    func_def_node = gast.FunctionDef(
        name=name,
        args=input_args,
        body=nodes,
        decorator_list=[],
        returns=None,
        type_comment=None)
    return func_def_node


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
    targets = [generate_name_node(return_name_ids, ctx=gast.Store())]
    assign_node = gast.Assign(targets=targets, value=cond_layer)

    return assign_node


def ast_to_func(ast_root, func_name, delete_on_exit=True):
    """
    Transform modified AST of decorated function into python callable object.
    """
    if not isinstance(ast_root, (gast.AST, ast.AST)):
        raise TypeError(
            "Type of ast_root should be gast.AST or ast.AST, but received %s." %
            type(ast_root))
    if isinstance(ast_root, gast.AST):
        ast_root = gast.gast_to_ast(ast_root)
    source = astor.to_source(ast_root)
    if six.PY2:
        source = source.encode('utf-8')
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    else:
        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8')

    # TODO(Aurelius84): more elegant way to transform ast into callable object
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
