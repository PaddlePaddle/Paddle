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
import codegen
import ast
import atexit
import os
import tempfile
import six
import imp

dygraph_class_to_static_api = {
    "BatchNorm": "batch_norm",
    "BilinearTensorProduct": "bilinear_tensor_prod",
    "Conv2D": "conv2d",
    "Conv3D": "conv3d",
    "Conv2DTranspose": "conv2d_transpose",
    "Conv3DTranspose": "conv3d_transpose",
    "CosineDecay": "cosine_decay",
    "Embedding": "embedding",
    "ExponentialDecay": "exponential_decay",
    "FC": "fc",
    "GroupNorm": "group_norm",
    "GRUUnit": "gru_unit",
    "InverseTimeDecay": "inverse_time_decay",
    "LayerNorm": "layer_norm",
    "Linear": "fc",
    "NaturalExpDecay": "natural_exp_decay",
    "NCE": "nce",
    "NoamDecay": "noam_decay",
    "PiecewiseDecay": "piecewise_decay",
    "PolynomialDecay": "polynomial_decay",
    "Pool2D": "pool2d",
    "PRelu": "prelu",
}


def to_static_api(dygraph_class):
    if dygraph_class in dygraph_class_to_static_api:
        return dygraph_class_to_static_api[dygraph_class]
    else:
        raise NotImplementedError(
            "Paddle dygraph API {class_name} cannot be converted "
            "to static graph at present.")


def to_static_ast(node, dygraph_class, dygraph_args, dygraph_keywords):
    static_api = to_static_api(dygraph_class)

    node.func = ast.Attribute(
        attr=static_api,
        value=ast.Attribute(
            attr='layers', value=ast.Name(
                ctx=ast.Load(), id='fluid')))
    node.args.extend(dygraph_args)
    node.keywords.extend(dygraph_keywords)

    ast.fix_missing_locations(node)

    return node


def to_assign_node(ori_node):
    assert isinstance(ori_node, ast.Call)

    assign_api = ast.parse('fluid.layers.assign').body[0].value
    ori_node.func = assign_api

    return ori_node


def _is_paddle_dygraph_api(obj):
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith("paddle.fluid.dygraph")


def is_dygraph_api(node):
    assert isinstance(node, ast.Call)
    func_src = codegen.to_source(node.func)
    try:
        import paddle.fluid as fluid
        return eval("_is_paddle_dygraph_api({})".format(func_src))
    except NameError:
        return False


def parse_class(node):
    assert isinstance(node, ast.Call)
    paddle_class = node.func.attr  # str
    paddle_args = node.args
    paddle_keywords = node.keywords  # list
    return paddle_class, paddle_args, paddle_keywords


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

    # Todo
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
