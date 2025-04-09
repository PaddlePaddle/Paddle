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
def clip_grad_norm_(
    parameters: Iterable[Tensor] | Tensor,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Tensor:
    r"""Clips gradient norm of the iterable parameters.

    Norms are calculated together on all gradients, just as they are
    connected into one vector. The gradient will be modified in place.

    This API can only run in dynamic graph mode, not static graph mode.

    Args:
        parameters (Iterable[paddle.Tensor] or paddle.Tensor): Tensors or a single Tensor
            that will be normalized gradients
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be `inf` for
            infinity norm.
        error_if_nonfinite (bool): if True, throw an error if the total
            norm of the gradients from :attr:`parameters` is `nan`,
            `inf`, or `-inf`.

    Returns:
        Total norm of the parameter gradients (treated as a single vector).

    Example:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
            >>> max_norm = float(5.0)
            >>> linear = paddle.nn.Linear(in_features=10, out_features=10)
            >>> out = linear(x)
            >>> loss = paddle.mean(out)
            >>> loss.backward()

            >>> paddle.nn.utils.clip_grad_norm_(linear.parameters(), max_norm)

            >>> sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters())
            >>> sdg.step()
    """
    if not paddle.in_dynamic_mode():
        raise RuntimeError('this API can only run in dynamic mode.')

    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]

    support_norm_type = [float("inf"), 0, 1, 2]
    if norm_type not in support_norm_type:
        raise ValueError(f'norm_type only support {support_norm_type}')

    grads = [p.grad_ for p in parameters if p.grad_ is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return paddle.to_tensor(0.0)
    if norm_type == float("inf"):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = (
            norms[0] if len(norms) == 1 else paddle.max(paddle.stack(norms))
        )
    else:
        total_norm = paddle.linalg.norm(
            paddle.stack(
                [paddle.linalg.norm(g.detach(), norm_type) for g in grads]
            ),
            norm_type,
        )

    if error_if_nonfinite and paddle.logical_or(
        total_norm.isnan(), total_norm.isinf()
    ):
        raise RuntimeError(
            f'The total norm of {norm_type} order of the gradients from '
            '`parameters` is non-finite, so it cannot be clipped. In any case, '
            'disable this error and scale the gradient by non-finite norm, '
            'set `error_if_nonfinite=False`'
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: when the coef is clamped to 1, it is redundant to multiply the clamped coef, but this
    # avoids the `if clip_coef < 1:` condition.
    clip_coef_clamped = clip_coef.clip_(max=1.0)

    for _, p in enumerate(parameters):
        if p.grad_ is not None:
            p.grad_.multiply_(y=clip_coef_clamped)

    return total_norm
