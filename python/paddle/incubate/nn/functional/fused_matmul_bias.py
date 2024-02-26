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

from paddle import _C_ops, _legacy_C_ops
from paddle.base.layer_helper import LayerHelper
from paddle.framework import (
    in_dynamic_mode,
    in_pir_mode,
)
from paddle.tensor.linalg import matmul


def fused_matmul_bias(
    x, y, bias=None, transpose_x=False, transpose_y=False, name=None
):
    """
    Applies matrix multiplication of two tensors and then bias addition if provided.
    This method requires CUDA version >= 11.6.

    Args:
        x (Tensor): the first input Tensor to be multiplied.
        y (Tensor): the second input Tensor to be multiplied. Its rank must be 2.
        bias (Tensor, optional): the input bias Tensor. If it is None, no bias addition would
            be performed. Otherwise, the bias is added to the matrix multiplication result. Default: None.
        transpose_x (bool, optional): Whether to transpose :math:`x` before multiplication. Default: False.
        transpose_y (bool, optional): Whether to transpose :math:`y` before multiplication. Default: False.
        name (str, optional): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('fused_gemm_epilogue is only supported when CUDA version >= 11.6')
            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_matmul_bias

            >>> paddle.set_device('gpu')
            >>> x = paddle.randn([3, 5])
            >>> y = paddle.randn([4, 5])
            >>> bias = paddle.randn([5])
            >>> out = fused_matmul_bias(x, y, bias)
            >>> print(out.shape)
            [3, 5]
    """
    if bias is None:
        return matmul(x, y, transpose_x, transpose_y, name)
    if in_dynamic_mode():
        return _legacy_C_ops.fused_gemm_epilogue(
            x, y, bias, 'trans_x', transpose_x, 'trans_y', transpose_y
        )
    if in_pir_mode():
        out, _ = _C_ops.fused_gemm_epilogue(
            x, y, bias, transpose_x, transpose_y, "none"
        )
        return out

    helper = LayerHelper('fused_matmul_bias', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='fused_gemm_epilogue',
        inputs={'X': x, 'Y': y, 'Bias': bias},
        outputs={'Out': out},
        attrs={'trans_x': transpose_x, 'trans_y': transpose_y},
    )
    return out


def fused_linear(x, weight, bias=None, transpose_weight=False, name=None):
    """
    Fully-connected linear transformation operator. This method requires CUDA version >= 11.6.

    Args:
        x (Tensor): the input Tensor to be multiplied.
        weight (Tensor): the weight Tensor to be multiplied. Its rank must be 2.
        bias (Tensor, optional): the input bias Tensor. If it is None, no bias addition would
            be performed. Otherwise, the bias is added to the matrix multiplication result. Default: None.
        transpose_weight (bool, optional): Whether to transpose :math:`weight` before multiplication. Default: False.
        name (str, optional): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('fused_gemm_epilogue is only supported when CUDA version >= 11.6')
            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_linear

            >>> paddle.set_device('gpu')
            >>> x = paddle.randn([3, 4])
            >>> weight = paddle.randn([4, 5])
            >>> bias = paddle.randn([5])
            >>> out = fused_linear(x, weight, bias)
            >>> print(out.shape)
            [3, 5]
    """
    return fused_matmul_bias(x, weight, bias, False, transpose_weight, name)


def fused_linear_activation(
    x, y, bias, trans_x=False, trans_y=False, activation=None
):
    """
    Fully-connected linear and activation transformation operator. This method requires CUDA version >= 11.6.

    Args:
        x (Tensor): the input Tensor to be multiplied.
        y (Tensor): the weight Tensor to be multiplied. Its rank must be 2.
        bias (Tensor): the input bias Tensor, the bias is added to the matrix multiplication result.
        trans_x (bool, optional): Whether to transpose :math:`x` before multiplication.
        trans_y (bool, optional): Whether to transpose :math:`y` before multiplication.
        activation (str, optional): Activation function, Currently, the available activation functions are
            limited to "gelu" (Gaussian Error Linear Unit) and "relu" (Rectified Linear Unit).
            These activation functions are applied to the output of the bias add. Default: None.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('fused_gemm_epilogue is only supported when CUDA version >= 11.6')
            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> from paddle.incubate.nn.functional import fused_linear_activation

            >>> paddle.set_device('gpu')
            >>> x = paddle.randn([3, 4])
            >>> weight = paddle.randn([4, 5])
            >>> bias = paddle.randn([5])
            >>> out = fused_linear_activation(x, weight, bias)
            >>> print(out.shape)
            [3, 5]
    """
    if activation is None:
        activation = "none"

    if in_dynamic_mode():
        return _legacy_C_ops.fused_gemm_epilogue(
            x,
            y,
            bias,
            'trans_x',
            trans_x,
            'trans_y',
            trans_y,
            'activation',
            activation,
        )

    if in_pir_mode():
        out, _ = _C_ops.fused_gemm_epilogue(
            x,
            y,
            bias,
            trans_x,
            trans_y,
            activation,
        )
        return out

    helper = LayerHelper('fused_matmul_bias', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='fused_gemm_epilogue',
        inputs={'X': x, 'Y': y, 'Bias': bias},
        outputs={'Out': out},
        attrs={
            'trans_x': trans_x,
            'trans_y': trans_y,
            'activation': activation,
        },
    )

    return out
