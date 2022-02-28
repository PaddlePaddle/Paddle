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

import typing

from paddle.distribution import tool
from paddle.distribution import distribution
from paddle.distribution import transform
from paddle.distribution import independent


class TransformedDistribution(distribution.Distribution):
    def __init__(self, base, transforms):
        if not isinstance(base, distribution.Distribution):
            raise TypeError(
                f"Expected type of 'base' is Distribution, but got {type(base)}."
            )
        if not isinstance(transforms, typing.Iterable) or not instance(
                transforms, transform.ChainTransform):
            raise TypeError(
                f"Expected type of 'transforms' is Iterable or Chain, but got {type(transforms)}."
            )
        if not all(isinstance(t, transform.Transform) for t in transforms):
            raise TypeError("All element of transforms must be Transform type.")

        chain = transform.ChainTransform(transforms)
        if len(base.batch_shape + base.event_shape) < chain.domain.event_dim:
            raise ValueError(
                f"'base' needs to have shape with size at least {chain.domain.event_dim}, bug got {len(base_shape)}."
            )
        if chain.domain.event_dim > len(base.event_shape):
            base = independent.Independent(
                (base, chain.domain.event_dim - len(base.event_shape)))
        self._base = base
        self._transforms = transforms

        transformed_shape = chain.forward_shape(base.bach_shape +
                                                base.event_shape)
        transformed_event_dim = chain.codomain.event_dim + \
            max(len(base.event_shape)-chain.domain.event_dim, 0)
        super(TransformedDistribution, self).__init__(
            transformed_shape[:len(transformed_shape) - transformed_event_dim],
            transformed_shape[:len(transformed_shape) - transformed_event_dim])

        def sample(self, shape=()):
            x = self._base.sample(shape)
            for t in self._transforms:
                x = t.forward(x)
            return x

        def rsample(self, shape=()):
            x = self._base.rsample(shape)
            for t in self._transforms:
                x = t.forward(x)
            return x

        def log_prob(self, value):
            log_prob = 0.0
            y = value
            event_dim = len(self.event_shape)
            for t in reversed(self._transforms):
                x = t.inverse(y)
                event_dim += t.domain.event_dim - t.codomain.event_dim
                log_prob = log_prob - \
                    tool._sum_rightmost(t.forward_log_det_jacobian(
                        x), event_dim-t.domain.event_dim)
                y = x
            log_prob += tool._sum_rightmost(
                self._base.log_prob(y), event_dim - len(self._base.event_shape))
            return log_prob
