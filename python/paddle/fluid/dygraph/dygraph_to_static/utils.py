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

import inspect
import gast
import astor
import atexit
import os
import tempfile
import six
import imp

STATIC_API = "static_api"
TO_DELETE_ARGS = "to_delete_args"
dygraph_class_to_static_api = {
    "BatchNorm": {
        STATIC_API: "batch_norm",
        TO_DELETE_ARGS: ["num_channels", "trainable_statistics", "dtype"]
    },
    "BilinearTensorProduct": {
        STATIC_API: "bilinear_tensor_product",
        TO_DELETE_ARGS: ["input1_dim", "input2_dim", "output_dim", "dtype"]
    },
    "Conv2D": {
        STATIC_API: "conv2d",
        TO_DELETE_ARGS: ["num_channels", "dtype"]
    },
    "Conv3D": {
        STATIC_API: "conv3d",
        TO_DELETE_ARGS: ["num_channels", "dtype"]
    },
    "Conv2DTranspose": {
        STATIC_API: "conv2d_transpose",
        TO_DELETE_ARGS: ["num_channels", "dtype"]
    },
    "Conv3DTranspose": {
        STATIC_API: "conv3d_transpose",
        TO_DELETE_ARGS: ["num_channels", "dtype"]
    },
    "CosineDecay": {
        STATIC_API: "cosine_decay",
        TO_DELETE_ARGS: ["begin", "step", "dtype"]
    },
    "Embedding": {
        STATIC_API: "embedding",
        TO_DELETE_ARGS: []
    },
    "ExponentialDecay": {
        STATIC_API: "exponential_decay",
        TO_DELETE_ARGS: ["begin", "step", "dtype"]
    },
    "FC": "fc",
    "GroupNorm": {
        STATIC_API: "group_norm",
        TO_DELETE_ARGS: ["channels", "dtype"]
    },
    "GRUUnit": {
        STATIC_API: "gru_unit",
        TO_DELETE_ARGS: ["name_scope", "dtype"]
    },
    "InverseTimeDecay": {
        STATIC_API: "inverse_time_decay",
        TO_DELETE_ARGS: ["begin", "step", "dtype"]
    },
    "LayerNorm": {
        STATIC_API: "layer_norm",
        TO_DELETE_ARGS: ["normalized_shape", "dtype"]
    },
    "Linear": {
        STATIC_API: "fc",
        TO_DELETE_ARGS: ["input_dim", "output_dim", "dtype"]
    },
    "NaturalExpDecay": {
        STATIC_API: "natural_exp_decay",
        TO_DELETE_ARGS: ["begin", "step", "dtype"]
    },
    "NCE": {
        STATIC_API: "nce",
        TO_DELETE_ARGS: ["dim", "dtype"]
    },
    "NoamDecay": {
        STATIC_API: "noam_decay",
        TO_DELETE_ARGS: ["begin", "step", "dtype"]
    },
    "PiecewiseDecay": {
        STATIC_API: "piecewise_decay",
        TO_DELETE_ARGS: ["begin", "step", "dtype"]
    },
    "PolynomialDecay": {
        STATIC_API: "polynomial_decay",
        TO_DELETE_ARGS: ["begin", "step", "dtype"]
    },
    "Pool2D": {
        STATIC_API: "pool2d",
        TO_DELETE_ARGS: []
    },
    "PRelu": {
        STATIC_API: "prelu",
        TO_DELETE_ARGS: ["channel", "input_shape", "dtype"]
    },
    "SpectralNorm": {
        STATIC_API: "spectral_norm",
        TO_DELETE_ARGS: ["weight_shape", "dtype"]
    },
}


def _delete_keywords_from(node, deleted_keywords):
    assert isinstance(node, gast.Call)
    node.keywords = [k for k in node.keywords if k.arg not in deleted_keywords]


def to_static_api(dygraph_class):
    if dygraph_class in dygraph_class_to_static_api:
        return dygraph_class_to_static_api[dygraph_class]
    else:
        raise NotImplementedError(
            "Paddle dygraph API {class_name} cannot be converted "
            "to static graph at present.")


def _add_keywords_to(node, dygraph_api_name):
    assert isinstance(node, gast.Call)
    if dygraph_api_name is "Linear":
        changed = False
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "output_dim":
                ast_keyword.arg = "size"
                changed = True

        node.keywords.append(
            gast.keyword(
                arg="num_flatten_dims",
                value=gast.Constant(
                    value=-1, kind=None)))

    if dygraph_api_name is "BilinearTensorProduct":
        changed = False
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "output_dim":
                ast_keyword.arg = "size"
                changed = True

    if dygraph_api_name is "PRelu":
        changed = False
        for ast_keyword in node.keywords:
            if ast_keyword.arg == "input":
                ast_keyword.arg = "x"
                changed = True

        if not changed:
            # todo: args and keywords of static function should be set more accurately
            pass

    return


def is_to_variable(node):
    assert isinstance(node, gast.Call)
    if is_dygraph_api(node):
        api_name = node.func.attr
        return api_name is "to_variable"
    return False


def to_static_ast(node, dygraph_class, dygraph_args, dygraph_keywords):
    static_info = to_static_api(dygraph_class)
    static_api = static_info[STATIC_API]

    node.func = gast.Attribute(
        attr=static_api,
        ctx=gast.Load(),
        value=gast.Attribute(
            attr='layers',
            ctx=gast.Load(),
            value=gast.Name(
                ctx=gast.Load(), id='fluid', annotation=None,
                type_comment=None)))
    node.args.extend(dygraph_args)
    node.keywords.extend(dygraph_keywords)
    _add_keywords_to(node, dygraph_class)
    _delete_keywords_from(node, static_info.get(TO_DELETE_ARGS, []))

    gast.fix_missing_locations(node)

    return node


def to_assign_node(ori_node):
    assert isinstance(ori_node, gast.Call)

    assign_api = gast.parse('fluid.layers.assign').body[0].value
    ori_node.func = assign_api
    return ori_node


def _is_paddle_dygraph_api(obj):
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith("paddle.fluid.dygraph")


def is_dygraph_api(node):
    assert isinstance(node, gast.Call)
    func_src = astor.to_source(gast.gast_to_ast(node.func))
    try:
        import paddle.fluid as fluid
        return eval("_is_paddle_dygraph_api({})".format(func_src))
    except NameError:
        return False


def parse_class(node):
    assert isinstance(node, gast.Call)
    paddle_class = node.func.attr  # str
    paddle_args = node.args
    paddle_keywords = node.keywords  # list
    return paddle_class, paddle_args, paddle_keywords


def ast_to_func(ast_root, func_name, delete_on_exit=True):
    """
    Transform modified AST of decorated function into python callable object.
    """
    ast_root = gast.gast_to_ast(ast_root)
    source = astor.to_source(ast_root)
    if six.PY2:
        source = source.encode('utf-8')
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    else:
        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8')

    # Todo: A more elegant way to import fluid is needed
    import_str = "import paddle.fluid as fluid\n"

    with f:
        module_name = os.path.basename(f.name)
        f.write(import_str)
        f.write(source)

    if delete_on_exit:
        atexit.register(lambda: os.remove(f.name))

    module = imp.load_source(module_name, f.name)

    assert hasattr(module, func_name), \
        "Function: {} doesn't exist in the Module transformed from AST.".format(func_name)

    return getattr(module, func_name)


def func_node_from_class(class_node):
    """
    Only get the method `forward` of class. And modify the name of `forward` to
    class_node.name + '_' + new_node.name. eg: ConvLayer_forward
    """
    assert isinstance(class_node, gast.ClassDef)
    new_node = None
    for child_node in class_node.body:
        if child_node.name == "forward":
            new_node = child_node
    if new_node:
        # modify func name and delete arg self
        new_func_name = class_node.name + '_' + new_node.name
        new_node.name = new_func_name

        arg_list = new_node.args.args
        new_node.args.args = [arg for arg in arg_list if arg.id != "self"]
        return new_node
    else:
        raise ValueError("Class {class_name} must have the method 'forward'".
                         format(class_node.name))
