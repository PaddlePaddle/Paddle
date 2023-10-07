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


import paddle
from paddle.distribution import exponential_family


class Gamma(exponential_family.ExponentialFamily):
    r"""
    Gamma distribution parameterized by :attr:`concentration` (aka "alpha") and :attr:`rate` (aka "beta").

    The probability density function (pdf) is

    .. math::

        f(x; \alpha, \beta, x > 0) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1}e^{-\beta x}

        \Gamma(\alpha)=\int_{0}^{\infty} x^{\alpha-1} e^{-x} \mathrm{~d} x, (\alpha>0)

    Example::
        .. code-block:: python

            >>> m = Gamma(paddle.tensor([1.0]), paddle.tensor([1.0]))

    """

    def __init__(self, concentration, rate):
        self.concentration = concentration
        self.rate = rate
        super().__init__(self.concentration.shape)

    @property
    def mean(self):
        """Mean of gamma distribuion.

        Returns:
            Tensor: mean value.
        """
        return self.concentration / self.rate

    @property
    def variance(self):
        """Variance of gamma distribution.

        Returns:
            Tensor: variance value.
        """
        return self.concentration / self.rate.pow(2)

    def prob(self, value):
        """Probability density funciotn evaluated at value

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: Probability.
        """
        return paddle.exp(self.log_prob(value))

    def log_prob(self, value):
        """Log probability density function evaluated at value

        Args:
            value (Tensor): Value to be evaluated

        Returns:
            Tensor: Log probability.
        """
        return (
            self.concentration * paddle.log(self.rate)
            + (self.concentration - 1) * paddle.log(value)
            - self.rate * value
            - paddle.lgamma(self.concentration)
        )

    def entropy(self):
        """Entropy of gamma distribution

        Returns:
            Tensor: Entropy.
        """
        return (
            self.concentration
            - paddle.log(self.rate)
            + paddle.lgamma(self.concentration)
            + (1.0 - self.concentration) * paddle.digamma(self.concentration)
        )

    def _natural_parameters(self):
        return (self.concentration - 1, -self.rate)

    def _log_normalizer(self, x, y):
        return paddle.lgamma(x + 1) + (x + 1) * paddle.log(-y.reciprocal())
