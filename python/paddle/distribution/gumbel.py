# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numbers
import math
import numpy as np

from paddle.distribution.transformed_distribution import TransformedDistribution
from paddle.fluid import framework as framework

try:
    from collections.abc import Iterable
except:
    from collections import Iterable


class Gumbel(TransformedDistribution):
    r"""The Gumbel distribution with location `loc` and `scale` parameters.

    Mathematical details

    The probability density function (pdf) is

    .. math::

        pdf(x; mu, sigma) = exp(-(x - mu) / sigma - exp(-(x - mu) / sigma)) / sigma


    In the above equation:

    * :math:`loc = \mu`: is the mean.
    * :math:`scale = \sigma`: is the std.

    Args:
        loc(int|float|tensor): The mean of gumbel distribution.The data type is int, float, tensor.
        scale(int|float|tensor): The std of gumbel distribution.The data type is int, float, tensor.

    Examples:
        .. code-block:: python

          import paddle
          from paddle.distribution.gumbel import Gumbel

          # Gumbel distributed with loc=0, scale=1
          dist = Gumbel(paddle.full([0.0]), paddle.full([1.0]))
          dist.sample()
          # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [4.14814520])
          value = paddle.full([0.5])
          dist.prob(value)
          # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [0.33070430])
          dist.log_prob(value
          # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [-1.10653067])
          dist.cdf(value)
          # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [0.54523915])
          dist.entropy()
          # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [1.57721567])
          dist.rsample([2])
          # Tensor(shape=[2, 1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [[0.80463481], [0.91893655]])

    """

    def __init__(self, loc, scale):

        if not isinstance(loc, (numbers.Real, framework.Variable)):
            raise TypeError(
                f"Expected type of loc is Real|Variable, but got {type(loc)}")
        if not isinstance(scale, (numbers.Real, framework.Variable)):
            raise TypeError(
                f"Expected type of scale is Real|Variable, but got {type(scale)}"
            )

        if isinstance(loc, numbers.Real):
            self.loc = paddle.full(shape=(), fill_value=loc)

        if isinstance(scale, numbers.Real):
            self.scale = paddle.full(shape=(), fill_value=scale)

        if loc.shape != scale.shape:
            self.loc, self.scale = paddle.broadcast_tensors(
                [self.loc, self.scale])

        finfo = np.finfo(self.loc.dtype)
        self.base_dist = paddle.distribution.Uniform(
            paddle.full_like(self.loc, float(finfo.tiny)),
            paddle.full_like(self.loc, float(1 - finfo.eps)))

        super(Gumbel, self).__init__(self.base_dist, ())

    @property
    def mean(self):
        """Mean of distribution

        The mean is

        .. math::

            mean = \mu + \sigma * γ

        In the above equation:

        * :math:`loc = \mu`: is the location parameter.
        * :math:`scale = \sigma`: is the scale parameter.
        * :math:`γ`: is the euler's constant.

        Returns:
            Tensor: mean value.

        """
        return self.loc + self.scale * np.euler_gamma

    @property
    def variance(self):
        """Variance of distribution.

        The variance is

        .. math::

            variance = \sigma^2 * \pi^2 / 6

        In the above equation:

        * :math:`scale = \sigma`: is the scale parameter.

        Returns:
            Tensor: The variance value.

        """
        temp = paddle.full(math.pi * math.pi, self.scale.dtype)

        return paddle.pow(self.scale, 2) * temp / 6

    @property
    def stddev(self):
        """Standard deviation of distribution

        The standard deviation is

        .. math::

            stddev = \sqrt{\sigma^2 * \pi^2 / 6}

        In the above equation:
        * :math:`scale = \sigma`: is the scale parameter.

        Returns:
            Tensor: std value
        """
        return paddle.sqrt(self.variance)

    def prob(self, value):
        """Probability density/mass function

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: probability.The data type is same with value.

        """
        if type(value) != type(self.loc):
            raise TypeError('value type must be Tensor')

        y = (self.loc - value) / self.scale
        return paddle.exp(y - paddle.exp(y)) / self.scale

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: log probability.The data type is same with value.

        """
        return paddle.log(self.prob(value))

    def cdf(self, value):
        """Cumulative distribution function.
        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: cumulative probability of value.

        """
        if value.dtype != self.loc.dtype:
            value = paddle.cast(value, self.loc.dtype)

        return paddle.exp(-paddle.exp(-(value - self.loc) / self.scale))

    def entropy(self):
        """Entropy of Gumbel distribution.

        Returns:
            Entropy of distribution.

        """
        return paddle.log(self.scale) + 1 + np.euler_gamma

    def sample(self, shape):
        """Sample from ``Gumbel``.

        Args:
            shape (Sequence[int], optional): The sample shape. Defaults to ().

        Returns:
            Tensor: A tensor with prepended dimensions shape.The data type is float32.

        """
        if not isinstance(shape, Iterable):
            raise TypeError('sample shape must be Iterable object.')

        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape):
        """reparameterized sample
        Args:
            shape (Sequence[int]): 1D `int32`. Shape of the generated samples.

        Returns:
            Tensor: A tensor with prepended dimensions shape.The data type is float32.

        """
        exp_transform = paddle.distribution.ExpTransform()
        affine_tf1 = paddle.distribution.AffineTransform(
            paddle.full(0), -paddle.ones_like(self.scale))
        affine_tf2 = paddle.distribution.AffineTransform(self.loc, -self.scale)

        return affine_tf2.forward(
            exp_transform.inverse(
                affine_tf1.forward(
                    exp_transform.inverse(self._base.sample(shape)))))
