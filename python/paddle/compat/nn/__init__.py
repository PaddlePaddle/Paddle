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

from typing import TYPE_CHECKING

import paddle
from paddle import nn
from paddle.framework import (
    in_dynamic_mode,
)
from paddle.utils.decorator_utils import ForbidKeywordsDecorator

from . import functional  # noqa: F401

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing import (
        Size2,
    )


__all__ = [
    'Unfold',
]


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
