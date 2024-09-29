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

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from paddle.incubate.nn import functional as F
from paddle.nn import Layer

if TYPE_CHECKING:
    from paddle import Tensor


class FusedDropoutAdd(Layer):
    r"""
    Fused Dropout and Add.

    Parameters:
        p (float|int, optional): Probability of setting units to zero. Default: 0.5
        mode(str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train (default), upscale the output at training time

                                  - train: :math:`out = x \times \frac{mask}{(1.0 - p)} + y`
                                  - inference: :math:`out = x + y`

                               2. downscale_in_infer, downscale the output at inference

                                  - train: :math:`out = x \times mask + y`
                                  - inference: :math:`out = x \times (1.0 - p) + y`
        name (str, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - x: N-D tensor.
        - y: N-D tensor.
        - output: N-D tensor, the same shape as x.


    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.device.set_device('gpu')
            >>> from paddle.incubate.nn.layer.fused_dropout_add import FusedDropoutAdd

            >>> x = paddle.to_tensor([[1,2,3], [4,5,6]], dtype="float32")
            >>> y = paddle.to_tensor([[1,2,3], [4,5,6]], dtype="float32")

            >>> m = FusedDropoutAdd(p=0.5)

            >>> out = m(x, y)
    """

    def __init__(
        self,
        p: float = 0.5,
        mode: Literal[
            'upscale_in_train', 'downscale_in_infer'
        ] = "upscale_in_train",
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.p = p
        self.mode = mode
        self.name = name

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out = F.fused_dropout_add(
            x,
            y,
            p=self.p,
            training=self.training,
            mode=self.mode,
            name=self.name,
        )
        return out

    def extra_repr(self) -> str:
        name_str = f', name={self.name}' if self.name else ''
        return f'p={self.p}, mode={self.mode}{name_str}'
