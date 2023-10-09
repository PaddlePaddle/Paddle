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
from paddle.framework import LayerHelper, in_dynamic_mode


def weight_quantize(x, algo="weight_only_int8"):
    """
    Quantization function for weight_only and llm.int8's weight.

    Args:
        x (Tensor): The input Tensor to be quantized, the data type is float16 or bfloat16.
        algo (str|None): The algo that is x will be apply, must be one of 'weight_only_int8',
            'weight_only_int4' and 'llm.int8', default: 'weight_only_int8'.

    Returns:
        out (Tensor): The Tensor which is the quantitative results, the data type is the same as that of x.
        scale (Tensor): The scale Tensor which is the scale of pre-channel, the data type is float32.
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            from paddle.nn.quant import weight_quantize

            paddle.device.set_device("cpu")
            x = np.random.randn(64, 32).astype('float16')
            x = paddle.to_tensor(x, dtype=paddle.float16, place=paddle.CPUPlace())
            out, scale = weight_quantize(x, algo='weight_only_int8')
            print(out.shape) # [32, 64]
            print(scale.shape) # [32]
    """

    if in_dynamic_mode():
        return _C_ops.weight_quantize(x, algo)
    else:
        type = "weight_quantize"
        helper = LayerHelper(type, **locals())
        out = helper.create_variable_for_type_inference('int8')
        scale = helper.create_variable_for_type_inference('float')

        helper.append_op(
            type=type,
            inputs={"x": x},
            outputs={'out': out, "scale": scale},
            attrs={"algo": algo},
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
        x (Tensor): The first input Tensor to be multiplied, the data type is float16 or bfloat16.
        weight (Tensor): The second input Tensor to be multiplied. Its rank must be 2.
        bias (Tensor|None): The input bias Tensor. If it is None, no bias addition would
            be performed. Otherwise, The bias is added to the matrix multiplication result.
        weight_scale (Tensor|None): The input scale Tensor Provided to weight for dequantization. Its rank must be 1.
        weight_dtype(str): The dtype of  weight Tensor, must be one of 'int8', 'int4', Defaulted to 'int8'.
    Returns:
        Tensor: the output Tensor, the data type is the same as that of x.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.nn.quant import weight_only_linear

            x = paddle.cast(paddle.randn([1, 2, 64]), dtype='float16')
            weight = paddle.cast(paddle.randint(0, 127, [32, 64]), dtype='int8')
            scale = paddle.randn([32], dtype='float32')
            bias = paddle.cast(paddle.randn([32]), dtype='float16')
            if paddle.device.cuda.get_device_capability()[0] >= 8:
                out = weight_only_linear(x, weight, bias=bias, weight_scale=scale, weight_dtype='int8')
                print(out.shape) # [1, 2, 32]
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

        inputs = {
            'x': [x],
            'weight': [weight],
            'weight_scale': [weight_scale],
        }
        if bias:
            inputs["bias"] = [bias]
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
        x (Tensor): the first input Tensor to be multiplied, the data type is float16 or bfloat16.
        weight (Tensor): the second input Tensor to be multiplied. Its rank must be 2.
        bias (Tensor|None): the input bias Tensor. If it is None, no bias addition would
            be performed. Otherwise, the bias is added to the matrix multiplication result.
        weight_scale (Tensor|None): the input scale Tensor Provided to weight for dequantization. Its rank must be 1.
        threshold(float): The min value of outlier in activation, outlier's channel will be apply multiply with x.dtype.

    Returns:
        Tensor: the output Tensor, the data type is the same as that of x.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.nn.quant import llm_int8_linear

            x = paddle.cast(paddle.randn([1, 2, 64]), dtype='float16')
            weight = paddle.cast(paddle.randint(0, 127, [32, 64]), dtype='int8')
            scale = paddle.randn([32], dtype='float32')
            bias = paddle.cast(paddle.randn([32]), dtype='float16')
            if paddle.device.cuda.get_device_capability()[0] >= 8:
                out = llm_int8_linear(x, weight, bias=bias, weight_scale=scale, threshold=6.0)
                print(out.shape) # [1, 2, 32]
    """
    if in_dynamic_mode():
        out = _C_ops.llm_int8_linear(x, weight, bias, weight_scale, threshold)
        return out
    else:
        type = "llm_int8_linear"
        helper = LayerHelper(type, **locals())
        dtype = x.dtype

        inputs = {
            'x': [x],
            'weight': [weight],
            'weight_scale': [weight_scale],
        }
        if bias:
            inputs["bias"] = [bias]
        attrs = {'threshold': threshold}

        out = helper.create_variable_for_type_inference(dtype)

        helper.append_op(
            type=type,
            inputs=inputs,
            outputs={'out': out},
            attrs=attrs,
        )
        return out
