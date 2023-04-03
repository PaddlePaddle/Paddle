# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import typing

import paddle
import paddle.framework.dtype as dtypes
from paddle.fluid import framework

from .phi_ops_map import op_info, op_map


class PrimOption:
    def __init__(self):
        self.enable_prim = False

    def get_status(self):
        return self.enable_prim

    def set_status(self, flag):
        self.enable_prim = flag


prim_option = PrimOption()


@framework.static_only
def prim_enabled():
    """
    Note:
        **ONLY available in the static graph mode.**

    Shows whether the automatic differentiation mechanism based on
    automatic differential basic operators is ON. Defaults to OFF.

    Returns:
        flag(bool): Whether the automatic differentiation mechanism based on automatic differential basic operators is ON.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.incubate.autograd import enable_prim, disable_prim, prim_enabled

            paddle.enable_static()
            enable_prim()

            print(prim_enabled()) # True

            disable_prim()

            print(prim_enabled()) # False
    """
    return prim_option.get_status()


@framework.static_only
def enable_prim():
    """
    Note:
        **ONLY available in the static graph mode.**

    Turns ON automatic differentiation mechanism based on automatic
    differential basic operators.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.incubate.autograd import enable_prim, prim_enabled

            paddle.enable_static()
            enable_prim()

            print(prim_enabled()) # True
    """
    prim_option.set_status(True)


@framework.static_only
def disable_prim():
    """
    Note:
        **ONLY available in the static graph mode.**

    Turns OFF automatic differentiation mechanism based on automatic
    differential basic operators.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.incubate.autograd import enable_prim, disable_prim, prim_enabled

            paddle.enable_static()
            enable_prim()

            print(prim_enabled()) # True

            disable_prim()

            print(prim_enabled()) # False
    """
    prim_option.set_status(False)


INT_DTYPE_2_STRING = {
    int(0): 'bool',
    int(1): 'int16',
    int(2): 'int32',
    int(3): 'int64',
    int(4): 'float16',
    int(5): 'float32',
    int(6): 'float64',
    int(20): 'uint8',
    int(21): 'int8',
    int(23): 'complex64',
    int(24): 'complex128',
}


def get_var_block(block, names):
    assert isinstance(names, list)
    if len(names) == 0:
        return None
    elif len(names) == 1:
        return block.var(names[0])
    else:
        return [block.var(name) for name in names]


def get_input_var_list(op):
    if op.input_names is None:
        return []
    else:
        return [
            get_var_block(op.block, op.input(n)) for n in sorted(op.input_names)
        ]


def _solve_arg(item):
    if "=" not in item:
        res = item
    else:
        res = item.split('=')[0]
    [arg_type, arg_name] = res.strip().split()
    return arg_type.strip(), arg_name.strip()


def _get_attr_value(op, arg_type, arg_name):
    op_content = op_map[op.type]
    if "attrs" in op_content.keys() and arg_name in op_content["attrs"].keys():
        arg_name = op_content["attrs"][arg_name]

    # Note: in some cases, attrs may be optional , thus assign None. Such case must be recorded.

    if arg_name not in op.attr_names:
        return None
    else:
        if arg_type == "DataType":
            return dtypes.dtype(op.attr(arg_name))
        return op.attr(arg_name)


def _get_args_values(op, phi_name):
    "get attrs' values for api args' values"
    args = op_info[phi_name]
    args_list = args["args"].split(",")
    inputs = []
    attrs = []

    for item in args_list:
        arg_type, arg_name = _solve_arg(item)
        op_content = op_map[op.type]
        # IntArray and Scalar are special cases which may cause dynamic shape. In these case, tensor-relative types are removed in composite op.
        if arg_type in ("IntArray", "Scalar"):
            tensor_key = "int_array" if arg_type == "IntArray" else "scalar"
            if op_content.get(tensor_key):
                tensor_content = op_content[tensor_key].get(arg_name)
                if not tensor_content:
                    raise ValueError(
                        f'No value found for {arg_name} of {arg_type} type for operator {op.type}.'
                    )
                for item in ("tensor_name", "tensors_name"):
                    # name of intarray may differ from operator arg_name
                    arg_name_new = tensor_content.get(item)
                    if (
                        arg_name_new is not None
                        and arg_name_new in op.input_names
                        and get_var_block(op.block, op.input(arg_name_new))
                    ):
                        raise ValueError(
                            f"Tensor type of {arg_type} is not supported in composite op. Please set other type value of input arg {arg_name_new} for operator {op.type}."
                        )

        if arg_type in ("Tensor", "Tensor[]"):
            # assume Tensor type must belong to inputs
            if (
                "inputs" in op_content.keys()
                and arg_name in op_content["inputs"].keys()
            ):
                inputs.append(op_content["inputs"][arg_name])
            else:
                inputs.append(arg_name)
        else:
            attr_value = _get_attr_value(op, arg_type, arg_name)
            attrs.append(attr_value)

    return inputs, attrs


def prepare_python_api_arguments(op):
    """
    Generate all args inputs of composite op. Because inputs of composite op is
    the same as phi op desribed in ops.yaml. So we need to map origin op to phi op
    and then push input data and attrs of origin op to correspondng phi op.
    """
    if op.input_names is None:
        return []
    else:
        if op.type in op_map:
            phi_name = op_map[op.type]["phi_name"]
        else:
            phi_name = op.type
        inputs, attrs = _get_args_values(op, phi_name)
        res = []
        for item in inputs:
            if item in op.input_names:
                res.append(get_var_block(op.block, op.input(item)))
            else:
                # Note: in some cases, inputs may be optional, thus assign None. Such case must be recorded.
                res.append(None)
        if attrs:
            res.extend(attrs)
        return res


def get_output_var_list(op):
    if op.output_names is None:
        return []
    else:
        return [
            get_var_block(op.block, op.output(n))
            for n in sorted(op.output_names)
        ]


def map_output_for_composite(op):
    """origin op outputs must be mapped into outputs of composite rule. map info has been defined in op_compat.yaml"""
    origin_output_names = op.output_names
    if origin_output_names is None:
        return []
    else:
        name = op.type
        res = []
        if op_map[name].get("outputs"):
            for item in op_map[name]["outputs"].keys():
                origin_output_name = op_map[name]["outputs"][item]
                if origin_output_name not in origin_output_names:
                    res.append(None)
                    # Note: in some cases, some output of origin op is optional, so op name may not be in origin_output_names
                    continue
                origin_output_var = get_var_block(
                    op.block, op.output(origin_output_name)
                )
                res.append(origin_output_var)
        elif len(origin_output_names) == 1:
            # When origin output num is 1, map info is not needed.
            origin_output_var = get_var_block(
                op.block, op.output(origin_output_names[0])
            )
            res.append(origin_output_var)
        else:
            raise ValueError(
                "When replace op with composite rule, there must exist output map info from origin op to composite rule."
            )
        return res


def flatten(inp):
    if inp is None or isinstance(inp, paddle.fluid.framework.Variable):
        return [inp]
    flattened = []
    for part in inp:
        flattened += flatten(part)
    return flattened


def flatten_and_remove_none(inp):
    flattened = flatten(inp)
    return [var for var in flattened if var is not None]


def as_tensors(xs):
    if isinstance(xs, framework.Variable):
        return (xs,)
    elif isinstance(xs, typing.Sequence):
        return tuple(xs)
    else:
        return xs
