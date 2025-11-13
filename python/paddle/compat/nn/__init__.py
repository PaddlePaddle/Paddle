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

from math import sqrt
from typing import TYPE_CHECKING

import paddle
from paddle import nn
from paddle.framework import (
    in_dynamic_mode,
)
from paddle.utils.decorator_utils import ForbidKeywordsDecorator

from . import functional

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import (
        DTypeLike,
        PlaceLike,
        Size2,
    )


__all__ = ['Unfold', 'Linear']


class Unfold(nn.Unfold):
    """
    A compatible version of paddle.nn.Unfold:

    The keyword arguments are in non-plural forms, example: `kernel_size` instead of `kernel_sizes`. `padding` restricts the size of the input to be 1(int) or 2, Size4 is not allowed.

    All the input parameters allow `Tensor` or `pir.Value` as inputs, and will be converted to lists. Other aspects are the same. To use a more input-flexible version of Unfold, please refer to `paddle.nn.Unfold`.

    Args:
        kernel_size(int|list|tuple|Tensor): The size of convolution kernel, should be [k_h, k_w]
            or an integer k treated as [k, k].
        stride(int|list|tuple|Tensor, optional): The strides, should be [stride_h, stride_w]
            or an integer stride treated as [sride, stride]. For default, strides will be [1, 1].
        padding(int|list|tuple|Tensor, optional): The paddings of each dimension, should be
            a single integer or [padding_h, padding_w]. If [padding_h, padding_w] was given, it will expanded to
            [padding_h, padding_w, padding_h, padding_w]. If an integer padding was given,
            [padding, padding, padding, padding] will be used. By default, paddings will be 0.
        dilation(int|list|tuple|Tensor, optional): The dilations of convolution kernel, should be
            [dilation_h, dilation_w], or an integer dilation treated as [dilation, dilation].
            For default, it will be [1, 1].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.randn((100, 3, 224, 224))
            >>> unfold = paddle.compat.nn.Unfold(kernel_size=[3, 3])
            >>> result = unfold(x)
            >>> print(result.shape)
            [100, 27, 49284]
    """

    kernel_sizes: Size2
    dilations: Size2
    paddings: Size2
    strides: Size2

    @ForbidKeywordsDecorator(
        illegal_keys={"kernel_sizes", "dilations", "paddings", "strides"},
        func_name="paddle.compat.nn.Unfold",
        correct_name="paddle.nn.Unfold",
    )
    def __init__(
        self,
        kernel_size: Size2,
        dilation: Size2 = 1,
        padding: Size2 = 0,
        stride: Size2 = 1,
    ) -> None:
        super().__init__(kernel_size, dilation, padding, stride)

    def forward(self, input: Tensor) -> Tensor:
        def to_list_if_necessary(x, size_check=False):
            res = x
            if in_dynamic_mode() and isinstance(
                x, (paddle.pir.Value, paddle.Tensor)
            ):
                res = x.tolist()
            else:
                if not isinstance(x, (list, tuple, int)):
                    raise TypeError(
                        "paddle.compat.nn.Unfold does not allow paddle.Tensor or pir.Value as inputs in static graph mode."
                    )
            if size_check and isinstance(res, (list, tuple)) and len(res) > 2:
                raise ValueError(
                    f"The `padding` field of paddle.compat.nn.Unfold can only have size 1 or 2, now len={len(res)}. \nDid you mean to use paddle.nn.Unfold() instead?"
                )
            return res

        return nn.functional.unfold(
            input,
            kernel_sizes=to_list_if_necessary(self.kernel_sizes),
            strides=to_list_if_necessary(self.strides),
            paddings=to_list_if_necessary(self.paddings, size_check=True),
            dilations=to_list_if_necessary(self.dilations),
            name=self.name,
        )


class Linear(nn.Layer):
    r"""

    Python compatible fully-connected linear transformation layer. For each input :math:`X` ,
    the equation is:

    .. math::

        Out = XW^T + b

    where :math:`W` is the weight and :math:`b` is the bias.

    Linear layer takes only one multi-dimensional tensor as input with the
    shape :math:`[*, in\_features]` , where :math:`*` means any
    number of additional dimensions. It multiplies input tensor with the transpose
    of weight (a 2-D tensor of shape :math:`[out\_features, in\_features]` ) and
    produces an output tensor of shape :math:`[*, out\_features]` .
    If ``bias`` is not False, the bias (a 1-D tensor of
    shape :math:`[out\_features]` ) will be created and added to the output. At the
    end of the initialization, ``reset_parameters`` will be called to initialize
    the ``weight`` and ``bias`` (if available) randomly.

    Parameters:
        in_features (int):
            The number of input units.
        out_features (int):
            The number of output units.
        bias (bool): If True, the bias (a 1-D tensor of shape :math:`[out\_features]` ) will be created and
            added to the output. Default: True.
        device (PlaceLike): The device of the parameters created. Default: None,
            representing the default paddle device.
        dtype (DTypeLike): The dtype of the parameters created. Default: None, and is set by
            the default dtype of Linear (float32).

    Variables:
        weight (paddle.Tensor): learnable parameters of the module of shape :math:`[out\_features, in\_features]`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k` is :math:`\frac{1}{in\_features}`.
        bias (paddle.Tensor): learnable parameters of the module of shape :math:`[out\_features]`. If ``bias`` is True,
            the values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k` is :math:`\frac{1}{in\_features}`.

    Shape:
        - input: Multi-dimensional tensor with shape :math:`[*, in\_features]` . Its data types are float16, float32, float64 ,The default is float32 .
        - output: Multi-dimensional tensor with shape :math:`[*, out\_features]` . The data type is the same as the input .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(100)

            >>> # Define the linear layer.
            >>> linear = paddle.compat.nn.Linear(2, 4, bias=True)
            >>> print(linear.weight)
            Parameter containing:
            Tensor(shape=[4, 2], dtype=float32, place=Place(cpu), stop_gradient=False,
                   [[-0.49191639,  0.28120756],
                    [-0.17887023,  0.40572405],
                    [ 0.35139430,  0.45717543],
                    [-0.06135514, -0.21088189]])

            >>> print(linear.bias)
            Parameter containing:
            Tensor(shape=[4], dtype=float32, place=Place(cpu), stop_gradient=False,
                   [ 0.49166456, -0.06108528, -0.14973064,  0.31168410])

            >>> x = paddle.arange(6, dtype="float32").reshape([3, 2])
            >>> y = linear(x)
            >>> print(y)
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
                   [[ 0.77287209,  0.34463876,  0.30744481,  0.10080221],
                    [ 0.35145447,  0.79834640,  1.92458415, -0.44367185],
                    [-0.06996319,  1.25205410,  3.54172373, -0.98814595]])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    @ForbidKeywordsDecorator(
        illegal_keys={"weight_attr", "bias_attr", "name"},
        func_name="paddle.compat.nn.Linear",
        correct_name="paddle.nn.Linear",
    )
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: PlaceLike | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        super().__init__()
        self._dtype = (
            self._helper.get_default_dtype() if dtype is None else dtype
        )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self.create_parameter(
            shape=[out_features, in_features],
            attr=None,
            dtype=self._dtype,
            is_bias=False,
            device=device,
        )
        self.bias = None
        if bias:
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=None,
                dtype=self._dtype,
                is_bias=True,
                device=device,
            )
        # The same parameter initialization as PyTorch
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        return functional.linear.__wrapped__(  # bypass ForbidKeywordsDecorator
            input=input, weight=self.weight, bias=self.bias
        )

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """

        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            # nn.init._calculate_fan_in_and_fan_out(self.weight) for 2D array
            # is equivalent to returning (weight.shape[1], weight.shape[0])
            # TODO(heqianyue): use _calculate_fan_in_and_fan_out when available
            fan_in = self.weight.shape[1]
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
