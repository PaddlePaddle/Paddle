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

from paddle import _C_ops
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode


def quant_for_compress(x, layout="weight_only_int8"):
    return _C_ops.quant_for_compress(x, layout)


def quantized_matmul(
    x,
    weight,
    bias=None,
    weight_scale=None,
    quant_method="None",
    name=None,
):
    if in_dynamic_mode():
        if (
            quant_method == "weight_only_int8"
            or quant_method == "weight_only_int4"
        ):
            out = _C_ops.weight_only_matmul(
                x, weight, bias, weight_scale, quant_method
            )
        elif quant_method == "llm.int8":
            out = _C_ops.llm_int8_matmul(
                x,
                weight,
                bias,
                weight_scale,
            )
        elif quant_method == "None":
            out = _C_ops.attn_matmul(x, weight, bias, False)
        else:
            raise ValueError(
                "Unknown quant_method: '{}'. quant_method must be in ['weight_only_int8', 'weight_only_int4', 'llm.int8', 'None'].".format(
                    quant_method
                )
            )

        return out
    else:
        ops_dict = {
            'weight_only_int8': "weight_only_matmul",
            'weight_only_int4': "weight_only_matmul",
            'llm.int8': "llm_int8_matmul",
            'None': "attn_matmul",
        }
        type = ops_dict[quant_method]
        helper = LayerHelper(type, **locals())
        dtype = x.dtype

        check_variable_and_dtype(x, 'x', ['float16', 'bfloat16'], type)

        if type == "weight_only_matmul":
            check_variable_and_dtype(weight, 'weight', ['int8'], type)
            inputs = {
                'x': [x],
                'weight': [weight],
                'bias': [bias],
                'weight_scale': [weight_scale],
            }
            attrs = {'quant_method': quant_method}
        elif type == "llm_int8_matmul":
            check_variable_and_dtype(weight, 'weight', ['int8'], type)
            inputs = {
                'x': [x],
                'weight': [weight],
                'bias': [bias],
                'weight_scale': [weight_scale],
            }
            attrs = {}
        elif type == "attn_matmul":
            inputs = {
                'x': [x],
                'weight': [weight],
                'bias': [bias],
            }
            attrs = {'transpose_weight': True}
        else:
            raise ValueError(
                "Unknown quant_method: '{}'. quant_method must be in ['weight_only_int8', 'weight_only_int4', 'llm.int8', 'None'].".format(
                    quant_method
                )
            )
        out = helper.create_variable_for_type_inference(dtype)

        helper.append_op(
            type=type,
            inputs=inputs,
            outputs={'Out': out},
            attrs=attrs,
        )
        return out
