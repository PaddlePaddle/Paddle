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

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import _non_static_mode
from paddle.tensor.linalg import matmul
from paddle import _C_ops, _legacy_C_ops


def fused_matmul_bias(x,
                      y,
                      bias=None,
                      transpose_x=False,
                      transpose_y=False,
                      name=None):
    """
    Applies matrix multiplication of two tensors and then bias addition if provided.
    This method requires CUDA version >= 11.6.

    Args:
        x (Tensor): the first input Tensor to be multiplied.
        y (Tensor): the second input Tensor to be multiplied. Its rank must be 2.
        bias (Tensor|None): the input bias Tensor. If it is None, no bias addition would
            be performed. Otherwise, the bias is added to the matrix multiplication result.
        transpose_x (bool): Whether to transpose :math:`x` before multiplication.
        transpose_y (bool): Whether to transpose :math:`y` before multiplication.
        name(str|None): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.

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
    if bias is None:
        return matmul(x, y, transpose_x, transpose_y, name)
    if _non_static_mode():
        return _legacy_C_ops.fused_gemm_epilogue(x, y, bias, 'trans_x',
                                                 transpose_x, 'trans_y',
                                                 transpose_y)

    helper = LayerHelper('fused_matmul_bias', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='fused_gemm_epilogue',
                     inputs={
                         'X': x,
                         'Y': y,
                         'Bias': bias
                     },
                     outputs={'Out': out},
                     attrs={
                         'trans_x': transpose_x,
                         'trans_y': transpose_y
                     })
    return out


def fused_linear(x, weight, bias=None, transpose_weight=False, name=None):
    """
    Fully-connected linear transformation operator. This method requires CUDA version >= 11.6.

    Args:
        x (Tensor): the input Tensor to be multiplied.
        weight (Tensor): the weight Tensor to be multiplied. Its rank must be 2.
        bias (Tensor|None): the input bias Tensor. If it is None, no bias addition would
            be performed. Otherwise, the bias is added to the matrix multiplication result.
        transpose_weight (bool): Whether to transpose :math:`weight` before multiplication.
        name(str|None): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn.functional import fused_linear

            x = paddle.randn([3, 4])
            weight = paddle.randn([4, 5])
            bias = paddle.randn([5])
            out = fused_linear(x, weight, bias)
            print(out.shape) # [3, 5]
    """
    return fused_matmul_bias(x, weight, bias, False, transpose_weight, name)
