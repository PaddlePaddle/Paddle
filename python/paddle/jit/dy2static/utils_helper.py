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

import astor
import numpy as np  # noqa: F401

import paddle  # noqa: F401
from paddle import fluid  # noqa: F401
from paddle.fluid import dygraph  # noqa: F401
from paddle.fluid import layers  # noqa: F401
from paddle.fluid.dygraph import to_variable  # noqa: F401
from paddle.utils import gast

from .ast_utils import ast_to_source_code
from .logging_utils import warn


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
DYGRAPH_MODULE_PREFIX = 'paddle.fluid.dygraph'


def is_dygraph_api(node):

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

    func_str = astor.to_source(gast.gast_to_ast(func_node)).strip()
    try:
        import paddle.jit.dy2static as _jst  # noqa: F401
        from paddle import to_tensor  # noqa: F401

        return eval(f"_is_api_in_module_helper({func_str}, '{module_prefix}')")
    except Exception:
        return False


def _is_api_in_module_helper(obj, module_prefix):
    m = inspect.getmodule(obj)
    return m is not None and m.__name__.startswith(module_prefix)


# Is numpy_api cannot reuse is_api_in_module because of numpy module problem
def is_numpy_api(node):
    assert isinstance(node, gast.Call), "Input non-Call node for is_numpy_api"
    func_str = astor.to_source(gast.gast_to_ast(node.func))
    try:
        module_result = eval(
            "_is_api_in_module_helper({}, '{}')".format(func_str, "numpy")
        )
        # BUG: np.random.uniform doesn't have module and cannot be analyzed
        # TODO: find a better way
        return module_result or (
            func_str.startswith("numpy.") or func_str.startswith("np.")
        )
    except Exception:
        return False


def is_paddle_api(node):
    return is_api_in_module(node, PADDLE_MODULE_PREFIX)


class NodeVarType:
    """
    Enum class of python variable types. We have to know some variable types
    during compile time to transfer AST. For example, a string variable and a
    tensor variable in if clause may lead to different conversion from dygraph
    to static graph.
    """

    ERROR = -1  # Returns when static analysis gets error
    UNKNOWN = 0  # Reserve for AST nodes have not known the type
    STATEMENT = 1  # For nodes representing statement (non-variable type)
    CALLABLE = 2

    # python data types
    NONE = 100
    BOOLEAN = 101
    INT = 102
    FLOAT = 103
    STRING = 104
    TENSOR = 105
    NUMPY_NDARRAY = 106

    # python collections
    LIST = 200
    SET = 201
    DICT = 202

    PADDLE_DYGRAPH_API = 300
    PADDLE_CONTROL_IF = 301
    PADDLE_CONTROL_WHILE = 302
    PADDLE_CONTROL_FOR = 303
    # Paddle API may not be visible to get source code.
    # We use this enum value to denote the type return by a Paddle API
    PADDLE_RETURN_TYPES = 304

    # If node.node_var_type in TENSOR_TYPES, it can be considered as tensor-dependent.
    TENSOR_TYPES = {TENSOR, PADDLE_RETURN_TYPES}

    Annotation_map = {
        "Tensor": TENSOR,
        "paddle.Tensor": TENSOR,
        "int": INT,
        "float": FLOAT,
        "bool": BOOLEAN,
        "str": STRING,
    }

    @staticmethod
    def binary_op_output_type(in_type1, in_type2):
        if in_type1 == in_type2:
            return in_type1

        if in_type1 == NodeVarType.UNKNOWN:
            return in_type2
        if in_type2 == NodeVarType.UNKNOWN:
            return in_type1

        supported_types = [
            NodeVarType.BOOLEAN,
            NodeVarType.INT,
            NodeVarType.FLOAT,
            NodeVarType.NUMPY_NDARRAY,
            NodeVarType.TENSOR,
            NodeVarType.PADDLE_RETURN_TYPES,
        ]

        if in_type1 not in supported_types:
            return NodeVarType.UNKNOWN
        if in_type2 not in supported_types:
            return NodeVarType.UNKNOWN

        forbidden_types = [NodeVarType.NUMPY_NDARRAY, NodeVarType.TENSOR]
        if in_type1 in forbidden_types and in_type2 in forbidden_types:
            return NodeVarType.UNKNOWN
        return max(in_type1, in_type2)

    @staticmethod
    def type_from_annotation(annotation):
        annotation_str = ast_to_source_code(annotation).strip()
        if annotation_str in NodeVarType.Annotation_map:
            return NodeVarType.Annotation_map[annotation_str]

        # raise warning if not found
        warn("Currently we don't support annotation: %s" % annotation_str)
        return NodeVarType.UNKNOWN
