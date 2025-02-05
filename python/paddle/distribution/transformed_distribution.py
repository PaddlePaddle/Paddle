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
from __future__ import annotations

import typing
from typing import TYPE_CHECKING

from paddle.distribution import distribution, independent, transform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle.distribution.distribution import Distribution
    from paddle.distribution.transform import Transform


class TransformedDistribution(distribution.Distribution):
    r"""
    Applies a sequence of Transforms to a base distribution.

    Args:
        base (Distribution): The base distribution.
        transforms (Sequence[Transform]): A sequence of ``Transform`` .

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> from paddle.distribution import transformed_distribution

            >>> d = transformed_distribution.TransformedDistribution(
            ...     paddle.distribution.Normal(0., 1.),
            ...     [paddle.distribution.AffineTransform(paddle.to_tensor(1.), paddle.to_tensor(2.))]
            ... )

            >>> # doctest: +SKIP('random sample')
            >>> print(d.sample([10]))
            Tensor(shape=[10], dtype=float32, place=Place(cpu), stop_gradient=True,
                [ 3.22699189,  1.12264419,  0.50283587,  1.83812487, -2.00740123,
                -2.70338631,  1.26663208,  4.47909021, -0.11529565,  4.32719326])
            >>> print(d.log_prob(paddle.to_tensor(0.5)))
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                -1.64333570)
            >>> # doctest: -SKIP
    """

    base: Distribution
    transforms: Sequence[Transform]

    def __init__(
        self, base: Distribution, transforms: Sequence[Transform]
    ) -> None:
        if not isinstance(base, distribution.Distribution):
            raise TypeError(
                f"Expected type of 'base' is Distribution, but got {type(base)}."
            )
        if not isinstance(transforms, typing.Sequence):
            raise TypeError(
                f"Expected type of 'transforms' is Sequence[Transform] or Chain, but got {type(transforms)}."
            )
        if not all(isinstance(t, transform.Transform) for t in transforms):
            raise TypeError("All element of transforms must be Transform type.")

        chain = transform.ChainTransform(transforms)
        base_shape = base.batch_shape + base.event_shape
        self._base = base
        self._transforms = transforms
        if not transforms:
            super().__init__(base.batch_shape, base.event_shape)
            return
        if len(base.batch_shape + base.event_shape) < chain._domain.event_rank:
            raise ValueError(
                f"'base' needs to have shape with size at least {chain._domain.event_rank}, bug got {len(base_shape)}."
            )
        if chain._domain.event_rank > len(base.event_shape):
            base = independent.Independent(
                base, chain._domain.event_rank - len(base.event_shape)
            )

        transformed_shape = chain.forward_shape(
            base.batch_shape + base.event_shape
        )
        transformed_event_rank = chain._codomain.event_rank + max(
            len(base.event_shape) - chain._domain.event_rank, 0
        )
        super().__init__(
            transformed_shape[
                : len(transformed_shape) - transformed_event_rank
            ],
            transformed_shape[
                len(transformed_shape) - transformed_event_rank :
            ],
        )

    def sample(self, shape: Sequence[int] = []) -> Tensor:
        """Sample from ``TransformedDistribution``.

        Args:
            shape (Sequence[int], optional): The sample shape. Defaults to [].

        Returns:
            [Tensor]: The sample result.
        """
        x = self._base.sample(shape)
        for t in self._transforms:
            x = t.forward(x)
        return x

    def rsample(self, shape: Sequence[int] = []) -> Tensor:
        """Reparameterized sample from ``TransformedDistribution``.

        Args:
            shape (Sequence[int], optional): The sample shape. Defaults to [].

        Returns:
            [Tensor]: The sample result.
        """
        x = self._base.rsample(shape)
        for t in self._transforms:
            x = t.forward(x)
        return x

    def log_prob(self, value: Tensor) -> Tensor:
        """The log probability evaluated at value.

        Args:
            value (Tensor): The value to be evaluated.

        Returns:
            Tensor: The log probability.
        """
        log_prob = 0.0
        y = value
        event_rank = len(self.event_shape)
        for t in reversed(self._transforms):
            x = t.inverse(y)
            event_rank += t._domain.event_rank - t._codomain.event_rank
            log_prob = log_prob - _sum_rightmost(
                t.forward_log_det_jacobian(x), event_rank - t._domain.event_rank
            )
            y = x
        log_prob += _sum_rightmost(
            self._base.log_prob(y), event_rank - len(self._base.event_shape)
        )
        return log_prob


def _sum_rightmost(value: Tensor, n: int) -> Tensor:
    return value.sum(list(range(-n, 0))) if n > 0 else value
