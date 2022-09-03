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

try:
    from collections.abc import Iterable
except:
    from collections import Iterable

import numbers

import numpy as np
import paddle
from paddle.distribution import distribution
from paddle.fluid import framework as framework


class Laplace(distribution.Distribution):
    r"""
    Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.

    Args:
        loc (scalar|Tensor): The mean of the distribution.
        scale (scalar|Tensor): The scale of the distribution.
    
    Examples:
        .. code-block:: python

                        import paddle

                        m = paddle.distribution.Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                        m.sample()  # Laplace distributed with loc=0, scale=1
                        # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, 
                        # [3.68546247])

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
            loc = paddle.full(shape=(), fill_value=loc)

        if isinstance(scale, numbers.Real):
            scale = paddle.full(shape=(), fill_value=scale)

        if (len(scale.shape) > 0 or len(loc.shape) > 0) and (loc.dtype
                                                             == scale.dtype):
            self.loc, self.scale = paddle.broadcast_tensors([loc, scale])
        else:
            self.loc, self.scale = loc, scale

        super(Laplace, self).__init__(self.loc.shape)

    @property
    def mean(self):
        """Mean of distribution.
        Returns:
            Tensor: The mean value.
        """
        return self.loc

    @property
    def stddev(self):
        """Standard deviation.
        Returns:
            Tensor: The std value.
        """
        return (2**0.5) * self.scale

    @property
    def variance(self):
        """Variance of distribution.
        Returns:
            Tensor: The variance value.
        """
        return self.stddev.pow(2)

    def _validate_value(self, value):
        """Argument dimension check for distribution methods such as `log_prob`,
        `cdf` and `icdf`. 

        Args:
          value (Tensor|Scalar): The input value, which can be a scalar or a tensor.
        
        Returns:
          loc, scale, value: The broadcasted loc, scale and value, with the same dimension and data type.
        """
        if isinstance(value, numbers.Real):
            value = paddle.full(shape=(), fill_value=value)
        if value.dtype != self.scale.dtype:
            value = paddle.cast(value, self.scale.dtype)
        if len(self.scale.shape) > 0 or len(self.loc.shape) > 0 or len(
                value.shape) > 0:
            loc, scale, value = paddle.broadcast_tensors(
                [self.loc, self.scale, value])
        else:
            loc, scale = self.loc, self.scale

        return loc, scale, value

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor|Scalar): The input value, can be a scalar or a tensor.

        Returns:
          Tensor: The log probability, whose data type is same with value.

        Examples:
            .. code-block:: python
                            import paddle

                            m = paddle.distribution.Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            value = paddle.to_tensor([0.1])
                            m.log_prob(value) 
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [-0.79314721])

        """
        loc, scale, value = self._validate_value(value)
        log_scale = -paddle.log(2 * scale)

        return (log_scale - paddle.abs(value - loc) / scale)

    def entropy(self):
        """Entropy of Laplace distribution.

        Returns:
            The entropy of distribution.

        Examples:
            .. code-block:: python
                            import paddle

                            m = paddle.distribution.Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            m.entropy()
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [1.69314718])
        """
        return 1 + paddle.log(2 * self.scale)

    def cdf(self, value):
        """Cumulative distribution function.
        Args:
            value (Tensor): The value to be evaluated.

        Returns:
            Tensor: The cumulative probability of value.
        
        Examples:
            .. code-block:: python
                            import paddle

                            m = paddle.distribution.Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            value = paddle.to_tensor([0.1])
                            m.cdf(value)
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [0.54758132])
        """
        loc, scale, value = self._validate_value(value)
        iterm = (0.5 * (value - loc).sign() *
                 paddle.expm1(-(value - loc).abs() / scale))

        return 0.5 - iterm

    def icdf(self, value):
        """Inverse Cumulative distribution function.
        Args:
            value (Tensor): The value to be evaluated.

        Returns:
            Tensor: The cumulative probability of value.

        Examples:
            .. code-block:: python
                            import paddle

                            m = paddle.distribution.Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            value = paddle.to_tensor([0.1])
                            m.icdf(value)
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [-1.60943794])
        """
        loc, scale, value = self._validate_value(value)
        term = value - 0.5

        return (loc - scale * (term).sign() * paddle.log1p(-2 * term.abs()))

    def sample(self, shape=()):
        """Generate samples of the specified shape.

        Args:
            shape(tuple[int]): The shape of generated samples.

        Returns:
            Tensor: A sample tensor that fits the Laplace distribution.

        Examples:
            .. code-block:: python
                            import paddle

                            m = paddle.distribution.Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            m.sample()  # Laplace distributed with loc=0, scale=1
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [3.68546247])
        """
        if not isinstance(shape, Iterable):
            raise TypeError(
                f'Expected shape should be tuple[int], but got {type(shape)}')

        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape):
        """Reparameterized sample.

        Args:
            shape(tuple[int]): The shape of generated samples.

        Returns:
            Tensor: A sample tensor that fits the Laplace distribution.

        Examples:
            .. code-block:: python
                            import paddle

                            m = paddle.distribution.Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            m.rsample((1,))  # Laplace distributed with loc=0, scale=1
                            # Tensor(shape=[1, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [[0.04337667]])
        """

        eps = self._get_eps()
        shape = self._extend_shape(shape) or (1, )
        uniform = paddle.uniform(shape=shape,
                                 min=float(np.nextafter(-1, 1)) + eps / 2,
                                 max=1. - eps / 2,
                                 dtype=self.loc.dtype)

        if len(self.scale.shape) == 0 and len(self.loc.shape) == 0:
            loc, scale, uniform = paddle.broadcast_tensors(
                [self.loc, self.scale, uniform])
        else:
            loc, scale = self.loc, self.scale

        return (loc - scale * uniform.sign() * paddle.log1p(-uniform.abs()))

    def _get_eps(self):
        """
        Get the eps of certain data type.

        Note: 
            Since paddle.finfo is temporarily unavailable, we 
            use hard-coding style to get eps value.

        Returns:
            Float: An eps value by different data types.
        """
        eps = 1.19209e-07
        if (self.loc.dtype == paddle.float64
                or self.loc.dtype == paddle.complex128):
            eps = 2.22045e-16

        return eps

    def kl_divergence(self, other):
        """Calculate the KL divergence KL(self || other) with two Laplace instances.

        Args:
            other (Laplace): An instance of Laplace.

        Returns:
            Tensor: The kl-divergence between two laplace distributions.

        Examples:
            .. code-block:: python
                            import paddle
                            
                            m1 = paddle.distribution.Laplace(paddle.to_tensor([0.0]), paddle.to_tensor([1.0]))
                            m2 = paddle.distribution.Laplace(paddle.to_tensor([1.0]), paddle.to_tensor([0.5]))
                            m1.kl_divergence(m2)
                            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                            # [1.04261160])
        """

        var_ratio = other.scale / self.scale
        t = paddle.abs(self.loc - other.loc)
        term1 = ((self.scale * paddle.exp(-t / self.scale) + t) / other.scale)
        term2 = paddle.log(var_ratio)

        return term1 + term2 - 1
