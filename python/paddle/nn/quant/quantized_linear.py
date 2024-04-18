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

from paddle import _C_ops, version
from paddle.base.data_feeder import check_dtype
from paddle.base.framework import convert_np_dtype_to_dtype_
from paddle.device.cuda import get_device_capability
from paddle.framework import (
    LayerHelper,
    in_dynamic_or_pir_mode,
)


def _get_arch_info():
    # Get SMVersion from device.
    cuda_version = version.cuda()
    if cuda_version is not None and cuda_version != 'False':
        major, minor = get_device_capability()
        arch = int(major * 10 + minor)
        return arch
    else:
        raise ValueError(
            "Paddle is not compiled with CUDA, we cannot get SMVersion from device, please try to compile Paddle with CUDA"
        )


def weight_quantize(x, algo="weight_only_int8", arch=None, group_size=-1):
    """
    Quantization function for weight_only and llm.int8's weight.

    Args:
        x (Tensor): The input Tensor to be quantized, the data type is float16 or bfloat16.
        algo (str): The algo that is x will be apply, must be one of 'weight_only_int8',
            'weight_only_int4' and 'llm.int8', default: 'weight_only_int8'.
        arch (int): The compute arch for target device. For example, A100 is 80, v100 is 70, if you do not assign arch, we will get arch from your device, default: None.
        group_size (int): The group size for weight quantization. -1 stands for default per-channel mode. Currently only support 64 or 128.

    Returns:
        out (Tensor): The Tensor which is the quantitative results, the data type is int8, the shape is transposition of x.
        scale (Tensor): The scale Tensor which is the scale of pre-channel, the data type is float32.
    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('No testing required')
            >>> import paddle
            >>> from paddle.nn.quant import weight_quantize

            >>> paddle.seed(2023)
            >>> x = paddle.rand(shape=[64, 32], dtype=paddle.float16)
            >>> out, scale = weight_quantize(x, algo='weight_only_int8')
            >>> print(out.shape)
            [32, 64]
            >>> print(scale.shape)
            [32]
    """
    if arch is None:
        arch = _get_arch_info()

    assert (
        arch == 70 or arch == 80 or arch == 86 or arch == 75
    ), f"Currently weight_quantize only support SM70/75/80/86. but got {arch} "

    assert (
        group_size == -1 or group_size == 64 or group_size == 128
    ), f"Currently group_size only support -1/64/128. but got {group_size} "
    if in_dynamic_or_pir_mode():
        return _C_ops.weight_quantize(x, algo, arch, group_size)
    else:
        type = "weight_quantize"
        helper = LayerHelper(type, **locals())
        out = helper.create_variable_for_type_inference('int8')
        scale = helper.create_variable_for_type_inference('float')

        helper.append_op(
            type=type,
            inputs={"x": x},
            outputs={'out': out, "scale": scale},
            attrs={"algo": algo, "arch": arch, "group_size": group_size},
        )
        return (out, scale)


def weight_dequantize(
    x, scale, algo="weight_only_int8", out_dtype='float16', group_size=-1
):
    """
    Dequantization function for weight_only and llm.int8's weight.

    Args:
        x (Tensor): The input Tensor to be dequantized, the data type is int8.
        scale (Tensor): The scale Tensor which is the output of weight_quantize, the data type is float32.
        algo (str): The algo that is x will be apply, must be one of 'weight_only_int8',
            'weight_only_int4' and 'llm.int8', default: 'weight_only_int8'.
        out_dtype (str|np.dtype): The output Tensor's data type, must be one of 'float16' and 'bfloat16', default: 'float16'.

    Returns:
        out (Tensor): The Tensor which is the dequantitative results, the data type is float16 or bfloat16, the shape is transposition of x.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('No testing required')
            >>> import paddle
            >>> from paddle.nn.quant import weight_quantize, weight_dequantize

            >>> paddle.seed(2023)
            >>> x = paddle.rand(shape=[64, 32], dtype=paddle.float16)
            >>> out, scale = weight_quantize(x, algo='weight_only_int8')
            >>> x_dequant = weight_dequantize(out, scale)
    """
    assert (
        group_size == -1 or group_size == 64 or group_size == 128
    ), f"Currently group_size only support -1/64/128. but got {group_size} "

    check_dtype(
        out_dtype, 'out_dtype', ['float16', 'bfloat16'], 'weight_dequantize'
    )
    out_dtype = convert_np_dtype_to_dtype_(out_dtype)
    if in_dynamic_or_pir_mode():
        return _C_ops.weight_dequantize(x, scale, algo, out_dtype, group_size)
    else:
        type = "weight_dequantize"
        helper = LayerHelper(type, **locals())
        out = helper.create_variable_for_type_inference(out_dtype)

        helper.append_op(
            type=type,
            inputs={"x": x, "scale": scale},
            outputs={'out': out},
            attrs={
                "algo": algo,
                "out_dtype": out_dtype,
                "group_size": group_size,
            },
        )
        return out


def weight_only_linear(
    x,
    weight,
    bias=None,
    weight_scale=None,
    weight_dtype="int8",
    arch=None,
    group_size=-1,
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
        arch (int): The compute arch for target device. For example, A100 is 80, v100 is 70, if you do not assign arch, we will get arch from your device, default: None.
        group_size (int): The group size for weight quantization. -1 stands for default per-channel mode. Currently only support 64 or 128.
    Returns:
        Tensor: the output Tensor, the data type is the same as that of x.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('No testing required')
            >>> import paddle
            >>> from paddle.nn.quant import weight_only_linear

            >>> x = paddle.cast(paddle.randn([1, 2, 64]), dtype='float16')
            >>> weight = paddle.cast(paddle.randint(0, 127, [32, 64]), dtype='int8')
            >>> scale = paddle.randn([32], dtype='float32')
            >>> bias = paddle.cast(paddle.randn([32]), dtype='float16')
            >>> if paddle.device.cuda.get_device_capability()[0] >= 8:
            ...    out = weight_only_linear(x, weight, bias=bias, weight_scale=scale, weight_dtype='int8')
            ...    print(out.shape)
            [1, 2, 32]
    """
    if arch is None:
        arch = _get_arch_info()

    assert (
        arch == 70 or arch == 80 or arch == 86 or arch == 75
    ), f"Currently weight_quantize only support SM70/75/80/86. but got {arch} "
    assert (
        group_size == -1 or group_size == 64 or group_size == 128
    ), f"Currently weight_quantize only support group size of -1, 64 or 128. but got {group_size} "

    if in_dynamic_or_pir_mode():
        out = _C_ops.weight_only_linear(
            x, weight, bias, weight_scale, weight_dtype, arch, group_size
        )
        return out
    else:
        check_dtype(
            weight_dtype, 'weight_dtype', ['int8', 'int4'], 'weight_only_linear'
        )
        type = "weight_only_linear"
        helper = LayerHelper(type, **locals())
        dtype = x.dtype

        inputs = {
            'x': [x],
            'weight': [weight],
            'weight_scale': [weight_scale],
        }
        if bias is not None:
            inputs["bias"] = [bias]
        attrs = {
            'weight_dtype': weight_dtype,
            'arch': arch,
            'group_size': group_size,
        }

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

            >>> # doctest: +SKIP('No testing required')
            >>> import paddle
            >>> from paddle.nn.quant import llm_int8_linear

            >>> x = paddle.cast(paddle.randn([1, 2, 64]), dtype='float16')
            >>> weight = paddle.cast(paddle.randint(0, 127, [32, 64]), dtype='int8')
            >>> scale = paddle.randn([32], dtype='float32')
            >>> bias = paddle.cast(paddle.randn([32]), dtype='float16')
            >>> if paddle.device.cuda.get_device_capability()[0] >= 8:
            ...    out = llm_int8_linear(x, weight, bias=bias, weight_scale=scale, threshold=6.0)
            ...    print(out.shape)
            [1, 2, 32]
    """
    if in_dynamic_or_pir_mode():
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


def apply_per_channel_scale(x, scales):
    """
    Apply pre-quant per channel scale on activations

    Args:
        x (Tensor): Input tensor representing the activations, the data type can be float16 or bfloat16.
        scales(Tensor): Per-channel scale factors for pre-quantization. Data type should be compatible with x.

    Returns:
        out (Tensor): The Tensor which is the pre-quant results, the data type is compatible with x.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('No testing required')
            >>> import paddle
            >>> from paddle.nn.quant import apply_per_channel_scale

            >>> paddle.seed(2023)
            >>> x = paddle.rand(shape=[64, 32], dtype=paddle.float16)
            >>> scales = paddle.rand(shape=[32], dtype=paddle.float16)
            >>> out = apply_per_channel_scale(x, scales)
    """

    if in_dynamic_or_pir_mode():
        return _C_ops.apply_per_channel_scale(x, scales)
    else:
        type = "apply_per_channel_scale"
        helper = LayerHelper(type, **locals())
        out = helper.create_variable_for_type_inference(x.dtype)

        helper.append_op(
            type=type,
            inputs={"x": [x], "scales": [scales]},
            outputs={"out": out},
        )
        return out
