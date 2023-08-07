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


def quant_for_infer(x, layout="weight_only_int8"):
    """
    Quantization function for weight_only and llm.int8's weight.

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
            from paddle.incubate.nn.functional import quant_for_infer

            x = paddle.randn([3, 4])
            out, scale = quant_for_infer(x, 'weight_only_int8')
            print(out.shape) # [4, 3]
            print(scale.shape) # [4]
    """

    if in_dynamic_mode():
        return _C_ops.quant_for_infer(x, layout)
    else:
        type = "quant_for_infer"
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


def weight_only_linear(
    x,
    weight,
    bias=None,
    weight_scale=None,
    weight_dtype="int8",
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
        quant_method(str): Method for doing quantized matrix multiplication, must be one of 'weight_only_int8', 'weight_only_int4', 'llm.int8', 'int8_matmul', default: 'weight_only_int8'.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn.functional import quantized_matmul

            x = paddle.randn([3, 4], dtype='float16')
            y = paddle.randn([5, 4], dtype='float16')
            weihgt, scale = quantized_matmul(x, y, layout='weight_only_int8')
            bias = paddle.randn([5])
            out = quantized_matmul(x, weihgt, bias=bias, weight_scale=scale, quant_method='weight_only_int8')
            print(out.shape) # [3, 5]
    """
    if in_dynamic_mode():
        out = _C_ops.weight_only_linear(
            x, weight, bias, weight_scale, weight_dtype
        )
        return out
    else:
        type = "weight_only_linear"
        helper = LayerHelper(type, **locals())
        dtype = x.dtype
        check_variable_and_dtype(x, 'x', ['float16', 'bfloat16'], type)
        check_variable_and_dtype(weight, 'weight', ['int8'], type)

        inputs = {
            'x': [x],
            'weight': [weight],
            'bias': [bias],
            'weight_scale': [weight_scale],
        }
        attrs = {'weight_dtype': weight_dtype}

        out = helper.create_variable_for_type_inference(dtype)

        helper.append_op(
            type=type,
            inputs=inputs,
            outputs={'out': out},
            attrs=attrs,
        )
        return out


def llm_int8_linear(
    x,
    weight,
    bias=None,
    weight_scale=None,
    threshold=6.0,
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
        quant_method(str): Method for doing quantized matrix multiplication, must be one of 'weight_only_int8', 'weight_only_int4', 'llm.int8', 'int8_matmul', default: 'weight_only_int8'.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn.functional import quantized_matmul

            x = paddle.randn([3, 4], dtype='float16')
            y = paddle.randn([5, 4], dtype='float16')
            weihgt, scale = quantized_matmul(x, y, layout='weight_only_int8')
            bias = paddle.randn([5])
            out = quantized_matmul(x, weihgt, bias=bias, weight_scale=scale, quant_method='weight_only_int8')
            print(out.shape) # [3, 5]
    """
    if in_dynamic_mode():
        out = _C_ops.llm_int8_linear(x, weight, bias, weight_scale, threshold)
        return out
    else:
        type = "llm_int8_linear"
        helper = LayerHelper(type, **locals())
        dtype = x.dtype
        check_variable_and_dtype(x, 'x', ['float16', 'bfloat16'], type)
        check_variable_and_dtype(weight, 'weight', ['int8'], type)

        inputs = {
            'x': [x],
            'weight': [weight],
            'bias': [bias],
            'weight_scale': [weight_scale],
        }
        attrs = {'threshold': threshold}

        out = helper.create_variable_for_type_inference(dtype)

        helper.append_op(
            type=type,
            inputs=inputs,
            outputs={'out': out},
            attrs=attrs,
        )
        return out
