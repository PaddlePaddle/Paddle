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
    """
    Quantization function for quantized_matmul API.

    Args:
        x (Tensor): the input Tensor to be quantized .
        layout (str|None): the layout of Tensor is quantized, must be one of 'weight_only_int8', 'weight_only_int4' and 'llm.int8'.

    Returns:
        out (Tensor): the Tensor which is the quantitative results.
        scale (Tensor): the scale Tensor which is the scale of pre-channel.
    Examples:
        .. code-block:: python

            # required: cpu
            import paddle
            from paddle.incubate.nn.functional import quant_for_compress

            x = paddle.randn([3, 4])
            out, scale = quant_for_compress(x, 'weight_only_int8')
            print(out.shape) # [4, 3]
            print(scale.shape) # [4]
    """

    if in_dynamic_mode():
        return _C_ops.quant_for_compress(x, layout)
    else:
        print("------------")
        type = "quant_for_compress"
        helper = LayerHelper(type, **locals())
        out = helper.create_variable_for_type_inference('int8')
        scale = helper.create_variable_for_type_inference('float')

        helper.append_op(
            type=type,
            inputs={"X": x},
            outputs={'Out': out, "Scale": scale},
            attrs={"layout": layout},
        )
        return (out, scale)


def quantized_matmul(
    x,
    weight,
    bias=None,
    weight_scale=None,
    quant_method="None",
):
    """
    Applies matrix multiplication of two tensors and then bias addition if provided.
    This method requires CUDA version >= 11.2.

    Args:
        x (Tensor): the first input Tensor to be multiplied.
        weight (Tensor): the second input Tensor to be multiplied. Its rank must be 2.
        bias (Tensor|None): the input bias Tensor. If it is None, no bias addition would
            be performed. Otherwise, the bias is added to the matrix multiplication result.
        weight_scale (Tensor|None): the input scale Tensor Provided to weight for dequantization. Its rank must be 1.
        quant_method(str): Method for doing quantized matrix multiplication, must be one of 'weight_only_int8', 'weight_only_int4', 'llm.int8', 'None', default: 'None'.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn.functional import fused_matmul_bias

            x = paddle.randn([3, 4])
            y = paddle.randn([4, 5])
            bias = paddle.randn([5])
            out = fused_matmul_bias(x, y, bias)
            print(out.shape) # [3, 5]
    """
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

        def get_ops(quant_method):
            if quant_method in ops_dict.keys():
                return ops_dict[quant_method]
            else:
                raise ValueError(
                    "Unknown quant_method: '{}'. quant_method must be in ['weight_only_int8', 'weight_only_int4', 'llm.int8', 'None'].".format(
                        quant_method
                    )
                )

        type = get_ops(quant_method)
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
            attrs = {'transpose_weight': False}

        out = helper.create_variable_for_type_inference(dtype)

        helper.append_op(
            type=type,
            inputs=inputs,
            outputs={'out': out},
            attrs=attrs,
        )
        return out
