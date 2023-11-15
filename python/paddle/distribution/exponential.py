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

import numbers

import numpy as np

import paddle
from paddle import distribution
from paddle.distribution import exponential_family


class Exponential(exponential_family.ExponentialFamily):
    r"""
    Exponential distribution parameterized by :attr:`rate`.

    The probability density function (pdf) is

    .. math::

        f(x; \theta) =  \theta e^{- \theta x },  (x \ge 0) $$

    Args:
        rate (float|Tensor): Rate parameter. The value of beta must be positive.

    Example::
        .. code-block:: python

            >>> import paddle

            >>> expon = paddle.distribution.Exponential(paddle.to_tensor([0.5]))
            >>> print(expon.mean)
            Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [2.])

            >>> print(expon.variance)
            Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [4.])

            >>> print(expon.entropy())
            Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [1.69314718])
    """

    def __init__(self, rate):
        if isinstance(rate, numbers.Real):
            rate = paddle.full(shape=(), fill_value=rate, dtype=paddle.float32)
        self.rate = rate
        super().__init__(self.rate.shape)

    @property
    def mean(self):
        """Mean of exponential distribuion.

        Returns:
            Tensor: mean value.
        """
        return self.rate.reciprocal()

    @property
    def variance(self):
        """Variance of exponential distribution.

        Returns:
            Tensor: variance value.
        """
        return self.rate.pow(-2)

    def sample(self, shape=()):
        """Generate samples of the specified shape.

        Args:
            shape (Sequence[int], optional): Shape of the generated samples.

        Returns:
            Tensor, A tensor with prepended dimensions shape.The data type is float32.
        """
        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=()):
        """Generate reparameterized samples of the specified shape.

        Args:
            shape (Sequence[int], optional): Shape of the generated samples.

        Returns:
            Tensor: A tensor with prepended dimensions shape.The data type is float32.
        """
        shape = distribution.Distribution._extend_shape(
            self, sample_shape=shape
        )

        uniform = paddle.uniform(
            shape=shape,
            min=float(np.finfo(dtype='float32').tiny),
            max=1.0,
            dtype=self.rate.dtype,
        )

        return -paddle.log(uniform) / self.rate

    def prob(self, value):
        """Probability density funciotn evaluated at value

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: Probability.
        """
        return self.rate * paddle.exp(-self.rate * value)

    def log_prob(self, value):
        """Log probability density function evaluated at value

        Args:
            value (Tensor): Value to be evaluated

        Returns:
            Tensor: Log probability.
        """
        return paddle.log(self.rate) - self.rate * value

    def entropy(self):
        """Entropy of exponential distribution

        Returns:
            Tensor: Entropy.
        """
        return 1.0 - paddle.log(self.rate)

    def kl_divergence(self, other):
        """The KL-divergence between two exponential distributions.

        Args:
            other (Exponential): instance of Exponential.

        Returns:
            Tensor: kl-divergence between two exponential distributions.
        """
        if not isinstance(other, Exponential):
            raise TypeError(
                f"Expected type of other is Exponential, but got {type(other)}"
            )

        rate_ratio = other.rate / self.rate
        t1 = -paddle.log(rate_ratio)
        return t1 + rate_ratio - 1

    @property
    def _natural_parameters(self):
        return (-self.rate,)

    def _log_normalizer(self, x):
        return -paddle.log(-x)
