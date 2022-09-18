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

import math
import warnings
import paddle
import numbers
import scipy

import numpy as np
from paddle.distribution import distribution
from paddle.distribution.uniform import Uniform
from paddle.distribution.transformed_distribution import TransformedDistribution
from paddle.distribution.transform import AffineTransform, ExpTransform
from paddle.fluid import framework as framework


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
        loc(int|float): The mean of normal distribution.The data type is int, float.
        scale(int|float): The std of normal distribution.The data type is int, float.

    Examples:
        .. code-block:: python

          import paddle
          from paddle.distribution import Gumbel

          # Define a single scalar Gumbel distribution.
          dist = Gumbel(loc=0., scale=1.)
    """

    def __init__(self, loc, scale):
        if not isinstance(loc, (numbers.Real, framework.Variable)):
            raise TypeError(
                f"Expected type of loc is Real|Variable, but got {type(loc)}")

        if not isinstance(scale, (numbers.Real, framework.Variable)):
            raise TypeError(
                f"Expected type of scale is Real|Variable, but got {type(scale)}"
            )
        self.loc, self.scale = paddle.broadcast_tensors([loc, scale])
        finfo = np.finfo(self.loc.dtype)
        if isinstance(loc, numbers.Number) and isinstance(scale, numbers.Number):
            base_dist = Uniform(finfo.tiny, 1 - finfo.eps)
        else:
            base_dist = Uniform(paddle.full_like(self.loc, finfo.tiny),
                                paddle.full_like(self.loc, 1 - finfo.eps))
        transforms = [ExpTransform().inverse(self), AffineTransform(loc=0, scale=-paddle.ones_like(self.scale)),
                      ExpTransform().inverse(self), AffineTransform(loc=loc, scale=-self.scale)]
        super(Gumbel, self).__init__(base_dist, transforms)

    @property
    def mean(self):
        """Mean of distribution

        The variance is

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
        return self.scale.pow(2) * math.pi.pow(2) / 6

    @property
    def stddev(self):
        """Standard deviation of distribution

        The standard deviation is

        .. math::

            stddev = \sqrt(\sigma^2 * \pi^2 / 6)

        In the above equation:

        * :math:`scale = \sigma`: is the scale parameter.

        Returns:
            Tensor: std value.

        """
        return math.sqrt(self.variance)

    def prob(self, value):
        """Probability density/mass function.

        The probability density is

        .. math::

            prob(value) = e^({y} - e^{y}) / \sigma

        .. math::

            y = (\mu - value) / \sigma

        In the above equation:

        * :math:`loc = \mu`: is the location parameter.
        * :math:`scale = \sigma`: is the scale parameter.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: probability.The data type is same with value.

        """
        y = (self.loc - value) / self.scale
        return math.exp(y - math.exp(y)) / self.scale

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability.The data type is same with value.

        """
        return math.log(self.prob(value))

    def entropy(self):
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            entropy(\sigma) = \\log(\sigma) + 1 + γ

        In the above equation:

        * :math:`scale = \sigma`: is the scale parameter.
        * :math:`γ`: is the euler's constant.

        Returns:
          Tensor: Shannon entropy of gumbel distribution.

        """
        return math.log(self.scale) + 1 + np.euler_gamma
