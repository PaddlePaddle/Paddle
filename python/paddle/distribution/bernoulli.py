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

from collections.abc import Iterable
from numbers import Number
import paddle
from paddle.distribution import distribution
from paddle.nn.functional import binary_cross_entropy_with_logits


class Bernoulli(distribution.Distribution):
    r"""
    Bernoulli distribution parameterized by :attr:`probs`.

    The probability mass function (PMF) for bernoulli is

    .. math::

        p(X=1)=p, p(X=0)=1-p


    Args:
        probs (Tensor): Probability of sampling '1'.

    Examples:

    .. code-block:: python

        import paddle

        multinomial = paddle.distribution.Bernoulli( paddle.to_tensor([0.1, 0.5, 0.7]))
        print(multinomial.sample((2, 3)))
        # Tensor(shape=[2, 3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
        #        [[[0., 0., 1.],
        #          [0., 1., 0.],
        #          [0., 1., 1.]],

        #         [[1., 0., 1.],
        #          [0., 0., 1.],
        #          [0., 1., 0.]]])
    """

    def __init__(self, probs):
        if probs.dim() < 1:
            raise ValueError(
                'probs parameter shoule not be none and over one dimension'
            )

        self.probs = probs
        self.logits = paddle.log(paddle.divide(self.probs, 1 - self.probs))
        batch_shape = self.probs.shape if not isinstance(probs, Number) else ()

        super().__init__(batch_shape)

    @property
    def mean(self):
        """mean of bernoulli distribuion.

        Returns:
            Tensor: mean value.
        """
        return self.probs

    @property
    def variance(self):
        """variance of bernoulli distribution.

        Returns:
            Tensor: variance value.
        """
        return self.probs * (1 - self.probs)

    def log_prob(self, value):
        """probability mass function evaluated at value

        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: probability of value.
        """
        if paddle.is_integer(value):
            value = paddle.cast(value, self.probs.dtype)

        logits, value = paddle.broadcast_tensors(
            [self.logits, value]
        )
        return -binary_cross_entropy_with_logits(logits, value, reduction='none')

    def sample(self, shape=()):
        """draw sample data from bernoulli distribution

        Args:
            sample_shape (tuple, optional): [description]. Defaults to ().
        """
        if not isinstance(shape, Iterable):
            raise TypeError('sample shape must be Iterable object.')

        shape = tuple(shape) + tuple(self.batch_shape) + tuple(self.event_shape)
        return paddle.bernoulli(self.probs.expand(shape))

    def entropy(self):
        """entropy of bernoulli distribution

        Returns:
            Tensor: entropy value
        """
        return binary_cross_entropy_with_logits(self.logits, self.probs, reduction='none')

