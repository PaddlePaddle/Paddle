# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


import inspect

import numpy as np  # noqa: F401

import paddle
from paddle import base  # noqa: F401
from paddle.base import dygraph, layers  # noqa: F401
from paddle.base.dygraph import to_variable  # noqa: F401
from paddle.utils import gast

from .ast_utils import ast_to_source_code


def index_in_list(array_list, item):
    try:
        return array_list.index(item)
    except ValueError:
        # Item not in array_list
        return -1


# Note(Aurelius): Do not forget the dot `.` to distinguish other
# module such as paddlenlp.
PADDLE_MODULE_PREFIX = 'paddle.'
DYGRAPH_TO_STATIC_MODULE_PREFIX = 'paddle.jit.dy2static'
DYGRAPH_MODULE_PREFIX = 'paddle.base.dygraph'


def is_dygraph_api(node):
    # TODO(SigureMo): Cleanup this function after we remove the BasicApiTransformer
    # Note: A api in module dygraph_to_static is not a real dygraph api.
    if is_api_in_module(node, DYGRAPH_TO_STATIC_MODULE_PREFIX):
        return False

    # TODO(liym27): A better way to determine whether it is a dygraph api.
    #  Consider the decorator @dygraph_only
    return is_api_in_module(node, DYGRAPH_MODULE_PREFIX)


def is_api_in_module(node, module_prefix):
    assert isinstance(node, gast.Call), "Input non-Call node for is_dygraph_api"

    # Python can have gast.Call as function, for example: covert_call(func)(x)
    # We only check the most outside function
    func_node = node.func
    while isinstance(func_node, gast.Call):
        func_node = func_node.func

    func_str = ast_to_source_code(func_node).strip()
    try:
        import paddle.jit.dy2static as _jst  # noqa: F401
        from paddle import to_tensor  # noqa: F401

        return eval(f"_is_api_in_module_helper({func_str}, '{module_prefix}')")
    except Exception:
        return False


def _is_api_in_module_helper(obj, module_prefix):
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith(module_prefix)


def is_paddle_api(node):
    return is_api_in_module(node, PADDLE_MODULE_PREFIX)


def set_dynamic_shape(variable, shape_list):
    if paddle.base.dygraph.base.in_to_static_mode():
        if isinstance(variable, paddle.base.framework.Variable):
            variable.desc.set_shape(shape_list)
        elif isinstance(variable, paddle.pir.Value):
            variable.set_shape(shape_list)
        else:
            raise TypeError(
                "In to_static mode, variable must be a Variable or Value"
            )
    else:
        # in dygraph mode, dynamic shape is not needed, just do nothing.
        return
