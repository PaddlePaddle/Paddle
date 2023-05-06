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

import paddle
from paddle.distribution import categorical, distribution


class Multinomial(distribution.Distribution):
    r"""
    Multinomial distribution parameterized by :attr:`total_count` and
    :attr:`probs`.

    In probability theory, the multinomial distribution is a generalization of
    the binomial distribution, it models the probability of counts for each side
    of a k-sided die rolled n times. When k is 2 and n is 1, the multinomial is
    the bernoulli distribution, when k is 2 and n is grater than 1, it is the
    binomial distribution, when k is grater than 2 and n is 1, it is the
    categorical distribution.

    The probability mass function (PMF) for multinomial is

    .. math::

        f(x_1, ..., x_k; n, p_1,...,p_k) = \frac{n!}{x_1!...x_k!}p_1^{x_1}...p_k^{x_k}

    where, :math:`n` is number of trials, k is the number of categories,
    :math:`p_i` denote probability of a trial falling into each category,
    :math:`{\textstyle \sum_{i=1}^{k}p_i=1}, p_i \ge 0`, and :math:`x_i` denote
    count of each category.

    Args:
        total_count (int): Number of trials.
        probs (Tensor): Probability of a trial falling into each category. Last
            axis of probs indexes over categories, other axes index over batches.
            Probs value should between [0, 1], and sum to 1 along last axis. If
            the value over 1, it will be normalized to sum to 1 along the last
            axis.

    Examples:

    .. code-block:: python

        import paddle

        multinomial = paddle.distribution.Multinomial(10, paddle.to_tensor([0.2, 0.3, 0.5]))
        print(multinomial.sample((2, 3)))
        # Tensor(shape=[2, 3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
        #        [[[1., 4., 5.],
        #          [0., 2., 8.],
        #          [2., 4., 4.]],

        #         [[1., 6., 3.],
        #          [3., 3., 4.],
        #          [3., 4., 3.]]])
    """

    def __init__(self, total_count, probs):
        if not isinstance(total_count, int) or total_count < 1:
            raise ValueError(
                'input parameter total_count must be int type and grater than zero.'
            )

        if probs.dim() < 1:
            raise ValueError(
                'probs parameter shoule not be none and over one dimension'
            )

        self.probs = probs / probs.sum(-1, keepdim=True)
        self.total_count = total_count
        self._categorical = categorical.Categorical(
            logits=self._probs_to_logits(probs)
        )

        super().__init__(probs.shape[:-1], probs.shape[-1:])

    @property
    def mean(self):
        """mean of multinomial distribuion.

        Returns:
            Tensor: mean value.
        """
        return self.probs * self.total_count

    @property
    def variance(self):
        """variance of multinomial distribution.

        Returns:
            Tensor: variance value.
        """
        return self.total_count * self.probs * (1 - self.probs)

    def prob(self, value):
        """probability mass function evaluated at value.

        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: probability of value.
        """
        return paddle.exp(self.log_prob(value))

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
            [paddle.log(self.probs), value]
        )
        logits[(value == 0) & (paddle.isinf(logits))] = 0

        return (
            paddle.lgamma(value.sum(-1) + 1)
            - paddle.lgamma(value + 1).sum(-1)
            + (value * logits).sum(-1)
        )

    def sample(self, shape=()):
        """draw sample data from multinomial distribution

        Args:
            sample_shape (tuple, optional): [description]. Defaults to ().
        """
        if not isinstance(shape, Iterable):
            raise TypeError('sample shape must be Iterable object.')

        samples = self._categorical.sample(
            [
                self.total_count,
            ]
            + list(shape)
        )
        return (
            paddle.nn.functional.one_hot(samples, self.probs.shape[-1])
            .cast(self.probs.dtype)
            .sum(0)
        )

    def entropy(self):
        """entropy of multinomial distribution

        Returns:
            Tensor: entropy value
        """
        n = paddle.full(
            shape=[], fill_value=self.total_count, dtype=self.probs.dtype
        )
        support = paddle.arange(
            self.total_count + 1, dtype=self.probs.dtype
        ).reshape((-1,) + (1,) * len(self.probs.shape))[1:]

        binomial_pmf = paddle.exp(self._binomial_logpmf(n, support))

        return (n * self._categorical.entropy() - paddle.lgamma(n + 1)) + (
            (binomial_pmf * paddle.lgamma(support + 1)).sum([0, -1])
        )

    def _binomial_logpmf(self, count, value):
        logits = self._probs_to_logits(self.probs, is_binary=True)

        factor_n = paddle.lgamma(count + 1)
        factor_k = paddle.lgamma(value + 1)
        factor_nmk = paddle.lgamma(count - value + 1)

        norm = (
            count * _clip_by_zero(logits)
            + count * paddle.log1p(paddle.exp(-paddle.abs(logits)))
            - factor_n
        )

        return value * logits - factor_k - factor_nmk - norm


def _binomial_support(count, dtype):
    return paddle.arange(count + 1, dtype=dtype)


def _clip_by_zero(x):
    # like clip(x, min=0) but grad at 0 is 0.5
    return (x.clip(min=0) + x - x.clip(max=0)) / 2
