# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import astor
import atexit
import copy
import gast
import imp
import inspect
import os
import six
import tempfile

dygraph_class_to_static_api = {
    "CosineDecay": "cosine_decay",
    "ExponentialDecay": "exponential_decay",
    "InverseTimeDecay": "inverse_time_decay",
    "NaturalExpDecay": "natural_exp_decay",
    "NoamDecay": "noam_decay",
    "PiecewiseDecay": "piecewise_decay",
    "PolynomialDecay": "polynomial_decay",
}


def _is_api_in_module_helper(obj, module_prefix):
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith(module_prefix)


def is_api_in_module(node, module_prefix):
    assert isinstance(node, gast.Call), "Input non-Call node for is_dygraph_api"
    func_str = astor.to_source(gast.gast_to_ast(node.func))
    try:
        import paddle.fluid as fluid
        import paddle
        return eval("_is_api_in_module_helper({}, '{}')".format(func_str,
                                                                module_prefix))
    except NameError:
        return False


def is_dygraph_api(node):
    return is_api_in_module(node, "paddle.fluid.dygraph")


def is_paddle_api(node):
    return is_api_in_module(node, "paddle.fluid")


# Is numpy_api cannot reuse is_api_in_module because of numpy module problem
def is_numpy_api(node):
    assert isinstance(node, gast.Call), "Input non-Call node for is_numpy_api"
    func_str = astor.to_source(gast.gast_to_ast(node.func))
    try:
        import numpy as np
        module_result = eval("_is_api_in_module_helper({}, '{}')".format(
            func_str, "numpy"))
        # BUG: np.random.uniform doesn't have module and cannot be analyzed
        # TODO: find a better way
        if not module_result:
            return func_str.startswith("numpy.") or func_str.startswith("np.")
    except NameError:
        return False


def _delete_keywords_from(node):
    assert isinstance(node, gast.Call)
    func_src = astor.to_source(gast.gast_to_ast(node.func))
    import paddle.fluid as fluid
    full_args = eval("inspect.getargspec({})".format(func_src))
    full_args_name = full_args[0]

    node.keywords = [k for k in node.keywords if k.arg in full_args_name]
    return


def to_static_api(dygraph_class):
    if dygraph_class in dygraph_class_to_static_api:
        return dygraph_class_to_static_api[dygraph_class]
    else:
        raise NotImplementedError("Paddle dygraph API {} cannot be converted "
                                  "to static graph at present.".format(
                                      dygraph_class))


def _add_keywords_to(node, dygraph_api_name):
    assert isinstance(node, gast.Call)
    if dygraph_api_name == "Linear":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "output_dim":
                ast_keyword.arg = "size"

        node.keywords.append(
            gast.keyword(
                arg="num_flatten_dims",
                value=gast.Constant(
                    value=-1, kind=None)))

    if dygraph_api_name == "BilinearTensorProduct":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "output_dim":
                ast_keyword.arg = "size"

    if dygraph_api_name == "PRelu":
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "input":
                ast_keyword.arg = "x"
    return


def is_to_variable(node):
    assert isinstance(node, gast.Call)
    if is_dygraph_api(node):
        api_name = node.func.attr
        return api_name == "to_variable"
    return False


def to_static_ast(node, class_node):
    assert isinstance(node, gast.Call)
    assert isinstance(class_node, gast.Call)
    static_api = to_static_api(class_node.func.attr)

    node.func = gast.Attribute(
        attr=static_api,
        ctx=gast.Load(),
        value=gast.Attribute(
            attr='layers',
            ctx=gast.Load(),
            value=gast.Name(
                ctx=gast.Load(), id='fluid', annotation=None,
                type_comment=None)))

    update_args_of_func(node, class_node, 'forward')

    node.args.extend(class_node.args)
    node.keywords.extend(class_node.keywords)
    _add_keywords_to(node, class_node.func.attr)
    _delete_keywords_from(node)

    gast.fix_missing_locations(node)

    return node


def to_assign_node(node):
    # Transform dygraph api `fluid.dygraph.to_variable` to static api `fluid.layers.assign`.
    # NOTE:
    #   1. Api `to_variable` supports data type {float16, float32, float64, int16, int32, int64, uint8, uint16},
    #   but api `assign` only supports {float32, float64, int32, int64, bool};
    #   2. If the input of api `assign` is numpy.ndarray, its size cannot be greater than 1024 * 1024.
    assert isinstance(node, gast.Call)
    assign_api = gast.parse('fluid.layers.assign').body[0].value
    node.func = assign_api

    if node.args:
        node.args = [node.args[0]]
        node.keywords = []
    else:
        for idx, kw in enumerate(node.keywords):
            if kw.arg == 'value':
                node.keywords[idx].arg = 'input'
                node.keywords = [node.keywords[idx]]
                node.args = []
                break
    return node


def update_args_of_func(node, dygraph_node, method_name):
    assert isinstance(node, gast.Call)
    if method_name not in ["__init__", "forward"]:
        raise ValueError(
            "The method name of class to update args should be '__init__' or 'forward'"
        )

    class_src = astor.to_source(gast.gast_to_ast(dygraph_node.func))
    import paddle.fluid as fluid
    if method_name == "__init__" or eval(
            "issubclass({}, fluid.dygraph.Layer)".format(class_src)):
        full_args = eval("inspect.getargspec({}.{})".format(class_src,
                                                            method_name))
        full_args_name = [
            arg_name for arg_name in full_args[0] if arg_name != "self"
        ]
    else:
        full_args_name = []
    added_keywords = []
    for idx, arg in enumerate(node.args):
        added_keywords.append(gast.keyword(arg=full_args_name[idx], value=arg))

    node.args = []
    node.keywords = added_keywords + node.keywords


def create_api_shape_node(tensor_shape_node):
    assert isinstance(tensor_shape_node, gast.Attribute)
    api_shape_node = gast.Call(
        func=gast.parse('fluid.layers.shape').body[0].value,
        args=[tensor_shape_node.value],
        keywords=[])
    return api_shape_node


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
                 "import paddle.fluid.layers as layers\n" \
                 "import numpy as np\n" \
                 "import numpy\n"
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
