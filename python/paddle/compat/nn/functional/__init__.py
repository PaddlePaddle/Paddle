# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import paddle
from paddle import _C_ops
from paddle.base.framework import Variable
from paddle.framework import (
    in_dynamic_mode,
)
from paddle.tensor import softmax
from paddle.utils.decorator_utils import ForbidKeywordsDecorator

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from paddle import Tensor
    from paddle._typing import (
        ShapeLike,
    )

    _PaddingTensorMode: TypeAlias = Literal[
        "zeros", "constant", "reflect", "replicate", "circular"
    ]


__all__ = ['pad', 'softmax', 'linear']


def _check_valid_pad_len(pad_len, x_dim, is_constant):
    if pad_len > 6 or pad_len < 0:
        raise ValueError(f"Expect len(pad) <= 6 and not -1, got: {pad_len}")
    max_dim = 2 * x_dim - (0 if is_constant else 2)
    if pad_len > max_dim:
        raise ValueError(
            f"len(pad) is bounded by input.ndim: expect len(pad) <= {max_dim}, got: {pad_len}"
        )


@ForbidKeywordsDecorator(
    illegal_keys={"x", "name", "data_format", "pad_from_left_axis"},
    func_name="paddle.compat.nn.functional.pad",
    correct_name="paddle.nn.functional.pad",
)
def pad(
    input: Tensor,
    pad: ShapeLike,
    mode: _PaddingTensorMode = 'constant',
    value: float = 0.0,
) -> Tensor:
    """

    PyTorch compatible version of :ref:`api_paddle_nn_functional_pad`. For the original API, see :ref:`api_paddle_nn_functional_pad` for more details.

    Pad tensor according to ``'pad'`` and ``'mode'``. All the padding operations under the hood starts from the **right** (last dim) of the tensor.

    Args:
        input (Tensor): The input tensor with data type float32, float64, int32, int64, complex64 or complex128.
        pad (Tensor|list[int]|tuple[int]): The padding size with data type int. Refer to Note for details.
        mode (str, optional): Four modes: ``'constant'`` (default), ``'reflect'``, ``'replicate'``, ``'circular'``. Default is ``'constant'``.

           - 'constant' mode, uses a constant value to pad the input tensor.
           - 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
           - 'replicate' mode, uses input boundaries to pad the input tensor.
           - 'circular' mode, uses circular input to pad the input tensor.

        value (float, optional): The value to fill the padded areas in 'constant' mode . Default is :math:`0.0`.

    Note:
        For non ``'constant'`` mode, padding size can not be greater than ``min(2 * input.ndim - 2, 6)``.
        Only 2D, 3D, 4D and 5D tensors are supported with up to the last 3 dims (if ndim >= 3) can be padded.

    Returns:
        Tensor, a Tensor padded according to pad and mode and data type is same as input.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input_shape = (1, 1, 3)
            >>> input_ = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
            >>> y = paddle.compat.nn.functional.pad(input_, [1, 0, 0, 1], value=0, mode='constant')
            >>> print(y)
            Tensor(shape=[1, 2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                [[[0., 1., 2., 3.],
                  [0., 0., 0., 0.]]])

            >>> # reflect 2D padding
            >>> input_ = paddle.arange(6).reshape([2, 3])
            >>> y = paddle.compat.nn.functional.pad(input=input_, pad=(1, 1), mode='reflect')
            >>> print(y)
            Tensor(shape=[2, 5], dtype=int64, place=Place(cpu), stop_gradient=True,
                [[1, 0, 1, 2, 1],
                 [4, 3, 4, 5, 4]])
    """

    assert mode in [
        'reflect',
        'replicate',
        'constant',
        'circular',
    ], (
        f"mode should be one of constant, reflect, replicate, circular, but got {mode}."
    )

    x_dim = len(input.shape)
    if in_dynamic_mode():
        if isinstance(pad, (Variable, paddle.Tensor)) and pad.size == 0:
            return input.clone()

    if (
        mode == "constant"
        and isinstance(pad, (list, tuple))
        and len(pad) != (x_dim - 2) * 2
    ):
        paddings = pad
        pad_value = value

        padding_len = len(paddings)
        # pad the length of paddings to 2*x_dim
        if padding_len < 2 * x_dim:
            pad_len_for_paddings = 2 * x_dim - padding_len
            paddings = paddings + ([0] if isinstance(pad, list) else (0,)) * (
                pad_len_for_paddings
            )

        # since the kernel pad from left axis, if we want to pad from right axis, we need to reverse the paddings
        paddings = [
            paddings[i - 1] if i % 2 == 1 else paddings[i + 1]
            for i in range(2 * x_dim - 1, -1, -1)
        ]
        pad_val = (
            pad_value
            if isinstance(pad_value, paddle.pir.Value)
            else float(pad_value)
        )
        return _C_ops.pad(input, paddings, pad_val)

    assert x_dim >= 1 and x_dim <= 5, (
        f"Input tensor dimension must be in [1-5] but got {x_dim}"
    )

    is_constant_mode = mode == 'constant'
    if (not is_constant_mode) and x_dim < 2:
        raise ValueError(
            f"Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now, got ndim: {x_dim}"
        )

    # pad the `pad` to be length = 6 (right padding), for example [1, 2] -> [1, 2, 0, 0, 0, 0]
    if isinstance(pad, (Variable, paddle.pir.Value)):
        pad_len = pad.shape[0]
        _check_valid_pad_len(pad_len, x_dim, is_constant_mode)
        pad = paddle.concat(
            [
                pad,
                paddle.zeros((6 - pad_len,), dtype="int32"),
            ],
            axis=0,
        )
    else:
        pad = list(pad)
        pad_len = len(pad)
        _check_valid_pad_len(pad_len, x_dim, is_constant_mode)
        pad.extend([0] * (6 - pad_len))

    ndim_to_unsqueeze = list(range(5 - x_dim))
    input = input.unsqueeze(axis=ndim_to_unsqueeze)

    out = _C_ops.pad3d(
        input,
        pad.tolist() if isinstance(pad, Variable) else pad,
        mode,
        value,
        "NCDHW",
    )
    if ndim_to_unsqueeze:
        return out.squeeze(axis=ndim_to_unsqueeze)
    return out


@ForbidKeywordsDecorator(
    illegal_keys={"x", "name"},
    func_name="paddle.compat.nn.functional.linear",
    correct_name="paddle.nn.functional.linear",
)
def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    r"""

    Fully-connected linear transformation operator. For each input :math:`x` ,
    the equation is:

    .. math::

        Out = xW^T + b

    where :math: `W` is the weight and :math:`b` is the bias.

    If the weight is a 2-D tensor of shape :math:`[out\_features, in\_features]` ,
    input should be a multi-dimensional tensor of shape
    :math:`[*, in\_features]` , where :math:`*` means any number of
    additional dimensions. The linear operator multiplies input tensor with
    weight and produces an output tensor of shape :math:`[*, out\_features]` ,
    If :math:`bias` is not None, the bias should be a 1-D tensor of shape
    :math:`[out\_features]` and will be added to the output.

    This implementation is aligned with PyTorch's linear function which computes
    :math:`y = xW^T + b`.

    Parameters:
        input (Tensor): Input tensor. The data type should be bfloat16, float16, float32 or float64.
            The input tensor should be of shape :math:`[*, in\_features]`, where :math:`*` means any number of additional dimensions, including none
        weight (Tensor): Weight tensor. The data type should be float16, float32 or float64.
            Shape should be [out_features, in_features].
        bias (Tensor, optional): Bias tensor. The data type should be float16, float32 or float64.
            If it is set to None, no bias will be added to the output units.

    Returns:
        Tensor, the shape is :math:`[*, out\_features]` and the
        data type is the same with input :math:`x` .

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.seed(2025)

            >>> x = paddle.arange(6, dtype=paddle.float32).reshape([3, 2])
            >>> x
            Tensor(shape=[3, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[0., 1.],
                    [2., 3.],
                    [4., 5.]])
            >>> weight = paddle.full(shape=[4, 2], fill_value=0.5, dtype="float32", name="weight")
            >>> weight
            Tensor(shape=[4, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[0.50000000, 0.50000000],
                    [0.50000000, 0.50000000],
                    [0.50000000, 0.50000000],
                    [0.50000000, 0.50000000]])
            >>> bias = paddle.ones(shape=[4], dtype="float32", name="bias")
            >>> y = paddle.compat.nn.functional.linear(x, weight, bias)
            >>> print(y)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[1.50000000, 1.50000000, 1.50000000, 1.50000000],
                    [3.50000000, 3.50000000, 3.50000000, 3.50000000],
                    [5.50000000, 5.50000000, 5.50000000, 5.50000000]])
    """
    # transpose y is True, since _C_ops.linear(input, weight.T, bias) can introduce more overhead. With CINN, matmul and add can be fused.
    out = _C_ops.matmul(input, weight, False, True)
    if bias is not None:
        out = _C_ops.add(out, bias)
    return out
