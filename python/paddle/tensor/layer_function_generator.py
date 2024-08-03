#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from paddle import _C_ops, _legacy_C_ops

from ..base.data_feeder import check_variable_and_dtype
from ..common_ops_import import Variable
from ..framework import (
    LayerHelper,
    OpProtoHolder,
    convert_np_dtype_to_dtype_,
    core,
    in_dynamic_mode,
    in_dynamic_or_pir_mode,
)

if TYPE_CHECKING:
    from paddle import Tensor

__all__ = []


def _convert_(name):
    """
    Formatting.

    Args:
       name: The name/alias

    This function takes in a name and converts it to a standard format of
    group1_group2. Where as per the regular expression, group1 can have
    alphabets and numbers and group2 has capital alphabets.

    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def generate_layer_fn(op_type: str):
    """Register the Python layer for an Operator.

    Args:
       op_type: The name of the operator to be created.

    This function takes in the operator type (sigmoid, mean , average etc) and
    creates the operator functionality.

    """
    op_proto = OpProtoHolder.instance().get_op_proto(op_type)
    not_intermediate_outputs = [
        output for output in op_proto.outputs if not output.intermediate
    ]
    intermediate_outputs = [
        output for output in op_proto.outputs if output.intermediate
    ]

    if len(not_intermediate_outputs) != 1:
        raise ValueError(
            "Only one non intermediate output operator can be"
            f"automatically generated. {op_type}"
        )

    if not_intermediate_outputs[0].duplicable:
        raise ValueError(
            "Only non duplicable op can be automatically generated."
        )

    for output in intermediate_outputs:
        if output.duplicable:
            raise ValueError(
                "The op can be automatically generated only when "
                "all intermediate ops are not duplicable."
            )

    o_name = not_intermediate_outputs[0].name
    intermediate_output_names = [output.name for output in intermediate_outputs]

    def infer_and_check_dtype(op_proto, *args, **kwargs):
        """
        This function performs the sanity check for dtype and
        instance type.
        """
        dtype = None
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            if len(val) == 0:
                if len(args) == 0:
                    continue
                val = [args[0]]
                args = args[1:]

            for each in val:
                if not isinstance(each, Variable):
                    raise ValueError(f"input of {op_type} must be variable")

                if dtype is None:
                    dtype = each.dtype
                elif dtype != each.dtype:
                    raise ValueError(
                        f"operator {op_type} must input same dtype. {dtype} vs {each.dtype}"
                    )

        if dtype is None:
            arg_dtype = kwargs.get("dtype")
            if arg_dtype:
                if not isinstance(arg_dtype, core.VarDesc.VarType):
                    dtype = convert_np_dtype_to_dtype_(arg_dtype)
                else:
                    dtype = arg_dtype
            else:
                dtype = core.VarDesc.VarType.FP32
        return dtype

    def func(*args, **kwargs) -> Tensor:
        helper = LayerHelper(op_type, **kwargs)

        dtype = infer_and_check_dtype(op_proto, *args, **kwargs)

        inputs = {}
        for ipt in op_proto.inputs:
            name = _convert_(ipt.name)
            val = kwargs.pop(name, [])
            if not isinstance(val, list) and not isinstance(val, tuple):
                val = [val]
            if len(val) == 0 and len(args) != 0:
                val = args[0]
                args = args[1:]
            inputs[ipt.name] = val

        outputs = {}
        out = kwargs.pop(_convert_(o_name), [])
        if out:
            out_var = out[0] if isinstance(out, (list, tuple)) else out
        else:
            out_var = helper.create_variable_for_type_inference(dtype=dtype)
        outputs[o_name] = [out_var]
        for name in intermediate_output_names:
            outputs[name] = [
                helper.create_variable_for_type_inference(dtype=dtype)
            ]
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=kwargs
        )
        return helper.append_activation(out_var)

    func.__name__ = op_type
    return func


def generate_activation_fn(op_type: str):
    """Register the Python layer for an Operator without Attribute.

    Args:
       op_type: The name of the operator to be created.

    This function takes in the operator type (sigmoid, exp , tanh etc) and
    creates the operator functionality.

    """

    def func(x, name: str | None = None) -> Tensor:
        if in_dynamic_or_pir_mode():
            if hasattr(_C_ops, op_type):
                op = getattr(_C_ops, op_type)
                return op(x)
            else:
                # TODO(dev): Because some ops' yaml has not been migrated.
                # Replace it with _C_ops while all yaml work is done.
                op = getattr(_legacy_C_ops, op_type)
                return op(x)
        else:
            if op_type not in ["abs", "exp", "square"]:
                check_variable_and_dtype(
                    x, 'x', ['float16', 'float32', 'float64'], op_type
                )
            else:
                # abs exp square ops support dtype(int32, int64, float16, float32, float64)
                check_variable_and_dtype(
                    x,
                    'x',
                    [
                        'int32',
                        'int64',
                        'float16',
                        'float32',
                        'float64',
                        'complex64',
                        'complex128',
                        'uint16',
                    ],
                    op_type,
                )

            helper = LayerHelper(op_type, **locals())

            output = helper.create_variable_for_type_inference(dtype=x.dtype)
            helper.append_op(
                type=op_type, inputs={"X": x}, outputs={"Out": output}
            )
            return output

    return func


def generate_inplace_fn(inplace_op_type):
    """Register the Python layer for an Inplace Operator without Attribute.

    Args:
       inplace_op_type: The name of the inplace operator to be created.

    This function takes in the inplace operator type (exp_ , ceil_ etc) and
    creates the operator functionality.
    """
    origin_op_type = inplace_op_type[:-1]

    def func(x, name=None):
        if in_dynamic_mode():
            if hasattr(_C_ops, inplace_op_type):
                op = getattr(_C_ops, inplace_op_type)
                return op(x)
            else:
                op = getattr(_legacy_C_ops, inplace_op_type)
                return op(x)

    func.__name__ = inplace_op_type
    func.__doc__ = f"""
Inplace version of ``{origin_op_type}`` API, the output Tensor will be inplaced with input ``x``.
Please refer to :ref:`api_paddle_{origin_op_type}`.
"""
    return func
