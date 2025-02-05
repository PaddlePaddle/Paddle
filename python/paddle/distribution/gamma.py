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

from typing import TYPE_CHECKING

import paddle
from paddle import distribution
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.framework import Variable
from paddle.distribution import exponential_family
from paddle.framework import in_dynamic_mode

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor, dtype


class Gamma(exponential_family.ExponentialFamily):
    r"""
    Gamma distribution parameterized by :attr:`concentration` (aka "alpha") and :attr:`rate` (aka "beta").

    The probability density function (pdf) is

    .. math::

        f(x; \alpha, \beta, x > 0) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1}e^{-\beta x}

        \Gamma(\alpha)=\int_{0}^{\infty} x^{\alpha-1} e^{-x} \mathrm{~d} x, (\alpha>0)

    Args:
        concentration (float|Tensor): Concentration parameter. It supports broadcast semantics.
            The value of concentration must be positive. When the parameter is a tensor,
            it represents multiple independent distribution with
            a batch_shape(refer to :ref:`api_paddle_distribution_Distribution`).
        rate (float|Tensor): Rate parameter. It supports broadcast semantics.
            The value of rate must be positive. When the parameter is tensor,
            it represent multiple independent distribution with
            a batch_shape(refer to :ref:`api_paddle_distribution_Distribution`).

    Example:
        .. code-block:: python

            >>> import paddle

            >>> # scale input
            >>> gamma = paddle.distribution.Gamma(0.5, 0.5)
            >>> print(gamma.mean)
            Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   1.)

            >>> print(gamma.variance)
            Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   2.)

            >>> print(gamma.entropy())
            Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   0.78375685)

            >>> # tensor input with broadcast
            >>> gamma = paddle.distribution.Gamma(paddle.to_tensor([0.2, 0.4]), paddle.to_tensor(0.6))
            >>> print(gamma.mean)
            Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [0.33333331, 0.66666663])

            >>> print(gamma.variance)
            Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [0.55555552, 1.11111104])

            >>> print(gamma.entropy())
            Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                   [-1.99634242,  0.17067254])
    """

    concentration: Tensor
    rate: Tensor
    dtype: dtype

    def __init__(
        self, concentration: float | Tensor, rate: float | Tensor
    ) -> None:
        if not in_dynamic_mode():
            check_type(
                concentration,
                'concentration',
                (float, Variable, paddle.pir.Value),
                'Gamma',
            )
            check_type(
                rate,
                'rate',
                (float, Variable, paddle.pir.Value),
                'Gamma',
            )

        # Get/convert concentration/rate to tensor.
        if self._validate_args(concentration, rate):
            self.concentration = concentration
            self.rate = rate
            self.dtype = convert_dtype(concentration.dtype)
        else:
            [self.concentration, self.rate] = self._to_tensor(
                concentration, rate
            )
            self.dtype = paddle.get_default_dtype()

        super().__init__(self.concentration.shape)

    @property
    def mean(self) -> Tensor:
        """Mean of gamma distribution.

        Returns:
            Tensor: mean value.
        """
        return self.concentration / self.rate

    @property
    def variance(self) -> Tensor:
        """Variance of gamma distribution.

        Returns:
            Tensor: variance value.
        """
        return self.concentration / self.rate.pow(2)

    def prob(self, value: float | Tensor) -> Tensor:
        """Probability density function evaluated at value

        Args:
            value (float|Tensor): Value to be evaluated.

        Returns:
            Tensor: Probability.
        """
        return paddle.exp(self.log_prob(value))

    def log_prob(self, value: float | Tensor) -> Tensor:
        """Log probability density function evaluated at value

        Args:
            value (float|Tensor): Value to be evaluated

        Returns:
            Tensor: Log probability.
        """
        return (
            self.concentration * paddle.log(self.rate)
            + (self.concentration - 1) * paddle.log(value)
            - self.rate * value
            - paddle.lgamma(self.concentration)
        )

    def entropy(self) -> Tensor:
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

    def sample(self, shape: Sequence[int] = []) -> Tensor:
        """Generate samples of the specified shape.

        Args:
            shape (Sequence[int], optional): Shape of the generated samples.

        Returns:
            Tensor, A tensor with prepended dimensions shape.The data type is float32.
        """
        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape: Sequence[int] = []) -> Tensor:
        """Generate reparameterized samples of the specified shape.

        Args:
            shape (Sequence[int], optional): Shape of the generated samples.

        Returns:
            Tensor: A tensor with prepended dimensions shape.The data type is float32.
        """
        shape = distribution.Distribution._extend_shape(
            self, sample_shape=shape
        )
        return paddle.standard_gamma(
            self.concentration.expand(shape)
        ) / self.rate.expand(shape)

    def kl_divergence(self, other: Gamma) -> Tensor:
        """The KL-divergence between two gamma distributions.

        Args:
            other (Gamma): instance of Gamma.

        Returns:
            Tensor: kl-divergence between two gamma distributions.
        """
        if not isinstance(other, Gamma):
            raise TypeError(
                f"Expected type of other is Exponential, but got {type(other)}"
            )

        t1 = other.concentration * paddle.log(self.rate / other.rate)
        t2 = paddle.lgamma(other.concentration) - paddle.lgamma(
            self.concentration
        )
        t3 = (self.concentration - other.concentration) * paddle.digamma(
            self.concentration
        )
        t4 = (other.rate - self.rate) * (self.concentration / self.rate)
        return t1 + t2 + t3 + t4

    def _natural_parameters(self) -> Tensor:
        return (self.concentration - 1, -self.rate)

    def _log_normalizer(self, x: Tensor, y: Tensor) -> Tensor:
        return paddle.lgamma(x + 1) + (x + 1) * paddle.log(-y.reciprocal())
