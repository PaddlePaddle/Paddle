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
from paddle.distribution.uniform import Uniform
from paddle.distribution.transformed_distribution import TransformedDistribution
from paddle.distribution.transform import AffineTransform, ExpTransform

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

        >>> example = Gumbel(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
        >>> m.sample()
        Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [4.14814520])

    """
    def __init__(self, loc, scale):

        self.batch_size_unknown = False
        self.all_arg_is_float = False
        self.loc = paddle.to_tensor(loc, dtype='float32')
        self.scale = paddle.to_tensor(scale, dtype='float32')

        finfo = np.finfo(dtype='float32')
        if isinstance(loc, numbers.Number) and isinstance(scale, numbers.Number):
            base_dist = Uniform(float(finfo.tiny), float(1 - finfo.eps))
        else:
            base_dist = Uniform(paddle.full_like(self.loc, float(finfo.tiny)),
                                paddle.full_like(self.loc, float(1 - finfo.eps)))

        super(Uniform, base_dist).__init__(self.loc.shape)
        self.transforms = [ExpTransform(),
                           AffineTransform(loc=paddle.to_tensor(0, dtype='float32'),
                                           scale=-paddle.ones_like(self.scale)),
                           ExpTransform(),
                           AffineTransform(loc=self.loc, scale=-self.scale)]
        super(Gumbel, self).__init__(base_dist, self.transforms)

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
    def mode(self):
        """Mode of distribution

        Returns:
            Tensor: mode value
        """
        return self.loc

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
        temp = paddle.to_tensor(math.pi * math.pi, dtype='float32')

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

        Examples:

            >>> example = Gumbel(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
            >>> value = paddle.to_tensor([0.5])
            >>> example.prob(value)
            Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            [0.33070430])

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

        Examples:

        >>> example = Gumbel(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
        >>> value = paddle.to_tensor([0.5])
        >>> example.prob(value
        Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
        [-1.10653067])

        """
        return paddle.log(self.prob(value))

    def cdf(self, value):
        """Cumulative distribution function.
        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: cumulative probability of value.

        Examples:

        >>> example = Gumbel(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
        >>> value = paddle.to_tensor([0.5])
        >>> example.cdf(value)
        Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [0.54523915])

        """
        if value.dtype != self.loc.dtype:
            value = paddle.cast(value, self.loc.dtype)

        return paddle.exp(-paddle.exp(- (value - self.loc) / self.scale))

    def entropy(self):
        """Entropy of Gumbel distribution.

        Returns:
            Entropy of distribution.

        Examples:

        >>> example = Gumbel(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
        >>> example.entropy()
        Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [1.57721567])

        """
        return paddle.log(self.scale) + 1 + np.euler_gamma

    def sample(self, shape=()):
        """Generate samples of the specified shape.

        Args:
          shape (list): 1D `int32`. Shape of the generated samples.

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.

        Examples:

        >>> example = Gumbel(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
        >>> m.sample()
        Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [4.14814520])

        """
        if not isinstance(shape, Iterable):
            raise TypeError('sample shape must be Iterable object.')

        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape):
        """reparameterized sample
        Args:
          shape (list): 1D `int32`. Shape of the generated samples.
          seed (int): Python integer number.

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.

        Examples:

        >>> example = Gumbel(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
        >>> example.rsample([2])
       Tensor(shape=[2, 1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[-0.00082598],
        [ 0.15536778]])

        """
        with paddle.no_grad():
            finfo = np.finfo(dtype='float32')

            if self.loc.dtype == paddle.float16 or \
                    self.loc.dtype == paddle.float32 or \
                    self.loc.dtype == paddle.complex64:
                eps = float(finfo.eps)
            else:
                raise TypeError("self.loc requires a floating point type")

            shape = self._extend_shape(shape)
            uniform_dist = paddle.uniform(shape=shape, min=eps - 1, max=1)

            return self.loc - self.scale * uniform_dist.sign() * paddle.log1p(-uniform_dist.abs())

    def _extend_shape(self, sample_shape):
        """compute shape of the sample

        Args:
            sample_shape (list or tuple): sample shape

        Returns:
            Tensor: generated sample data shape

        Examples:

        >>> example = Gumbel(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
        >>> example._extend_shape([2, 4])
        [2, 4, 1]

        """
        if self.batch_size_unknown:
            self._batch_shape = ()
        return list(sample_shape) + list(self._batch_shape)
