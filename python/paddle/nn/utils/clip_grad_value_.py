#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

if TYPE_CHECKING:
    from collections.abc import Iterable

    from paddle import Tensor

__all__ = []


@paddle.autograd.no_grad()
def clip_grad_value_(
    parameters: Iterable[Tensor] | Tensor,
    clip_value: float,
) -> None:
    r"""
    Clips gradient of an iterable of parameters at specified value.
    The gradient will be modified in place.
    This API can only run in dynamic graph mode, not static graph mode.

    Args:
        parameters (Iterable[paddle.Tensor]|paddle.Tensor): Tensors or a single Tensor
            that will be normalized gradients
        clip_value (float|int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`

    Example:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.uniform([10, 10], min=-10.0, max=10.0, dtype='float32')
            >>> clip_value = float(5.0)
            >>> linear = paddle.nn.Linear(in_features=10, out_features=10)
            >>> out = linear(x)
            >>> loss = paddle.mean(out)
            >>> loss.backward()
            >>> paddle.nn.utils.clip_grad_value_(linear.parameters(), clip_value)
            >>> sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters())
            >>> sdg.step()
    """
    if not paddle.in_dynamic_mode():
        raise RuntimeError('this API can only run in dynamic mode.')

    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]

    clip_value = float(clip_value)

    for _, p in enumerate(parameters):
        if p.grad is not None:
            p.grad.clip_(min=-clip_value, max=clip_value)
