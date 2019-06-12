# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

from . import control_flow
from . import tensor
from . import ops
from . import nn
import math
import numpy as np
__all__ = ['Uniform', 'Normal']


class Distribution(object):
    """
    Distribution is the abstract base class for probability distributions.
    """

    def sample(self):
        """Sampling from the distribution."""
        raise NotImplementedError

    def entropy(self):
        """The entropy of the distribution."""
        raise NotImplementedError

    def kl_divergence(self, other):
        """The KL-divergence between self distributions and other."""
        raise NotImplementedError

    def log_prob(self, value):
        """Log probability density/mass function."""
        raise NotImplementedError

    def _validate_args(self, *args):
        """
        Argument validation for distribution args
        Args:
            value (float, list, numpy.ndarray, Variable)
        Raises
            ValueError: if one argument is Variable, all arguments should be Variable
        """
        is_variable = False
        is_number = False
        for arg in args:
            if isinstance(arg, tensor.Variable):
                is_variable = True
            else:
                is_number = True

        if is_variable and is_number:
            raise ValueError(
                'if one argument is Variable, all arguments should be Variable')

        return is_variable

    def _to_variable(self, *args):
        """
        Argument convert args to Variable
        Args:
            value (float, list, numpy.ndarray, Variable)
        Returns:
            Variable of args.
        """
        numpy_args = []
        variable_args = []
        tmp = 0.

        for arg in args:
            valid_arg = False
            for cls in [float, list, np.ndarray, tensor.Variable]:
                if isinstance(arg, cls):
                    valid_arg = True
                    break
            assert valid_arg, "type of input args must be float, list, np.ndarray or Variable"
            if isinstance(arg, float):
                arg = np.zeros(1) + arg
            arg_np = np.array(arg).astype('float32')
            tmp = tmp + arg_np
            numpy_args.append(arg_np)
        for arg in numpy_args:
            arg_broadcasted, _ = np.broadcast_arrays(arg, tmp)
            arg_variable = tensor.create_tensor(dtype='float32')
            tensor.assign(arg_broadcasted, arg_variable)
            variable_args.append(arg_variable)

        return tuple(variable_args)


class Uniform(Distribution):
    """Uniform distribution with `low` and `high` parameters.
    #### Mathematical Details
    The probability density function (pdf) is,
    ```none
    pdf(x; a, b) = I[a <= x < b] / Z
    Z = b - a
    ```
    where
    - `low = a`,
    - `high = b`,
    - `Z` is the normalizing constant, and
    - `I[predicate]` is the [indicator function](
    https://en.wikipedia.org/wiki/Indicator_function) for `predicate`.
    The parameters `low` and `high` must be shaped in a way that supports
    broadcasting (e.g., `high - low` is a valid operation).
    #### Examples
    ```python
    # Without broadcasting:
    u1 = Uniform(low=3.0, high=4.0)  # a single uniform distribution [3, 4]
    u2 = Uniform(low=[1.0, 2.0],
               high=[3.0, 4.0])  # 2 distributions [1, 3], [2, 4]
    u3 = Uniform(low=[[1.0, 2.0],
                    [3.0, 4.0]],
               high=[[1.5, 2.5],
                     [3.5, 4.5]])  # 4 distributions
    ```
    ```python
    # With broadcasting:
    u1 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])  # 3 distributions
    ```
    ```python
    # Variable as input:
    low_array = np.array([1.0, 2.0])
    high_array = np.array([3.0, 4.0])
    low_variable = layers.create_tensor(dtype='float32')
    high_variable = layers.create_tensor(dtype='float32')
    layers.assign(low_array, low_variable)
    layers.assign(high_array, high_variable)
    u1 = Uniform(low=low_variable, high=high_variable)
    ```
    """

    def __init__(self, low, high):
        self.all_arg_is_float = False
        self.batch_size_unknown = False
        if self._validate_args(low, high):
            self.batch_size_unknown = True
            self.low = low
            self.high = high
        else:
            if isinstance(low, float) and isinstance(high, float):
                self.all_arg_is_float = True
            self.low, self.high = self._to_variable(low, high)

    def sample(self, shape, seed=0):
        """Generate samples of the specified shape.
        Args:
          shape: 1D `int32`. Shape of the generated samples.
          seed: Python integer seed for RNG
        Returns:
          samples: a `Variable` with prepended dimensions `shape`.
        """
        batch_shape = list((self.low + self.high).shape)
        if self.batch_size_unknown:
            output_shape = shape + batch_shape
            zero_tmp = tensor.fill_constant_batch_size_like(
                self.low + self.high, batch_shape + shape, 'float32', 0.)
            uniform_random_tmp = nn.uniform_random_batch_size_like(
                zero_tmp, zero_tmp.shape, min=0., max=1., seed=seed)
            output = uniform_random_tmp * (zero_tmp + self.high - self.low
                                           ) + self.low
            return nn.reshape(output, output_shape)
        else:
            output_shape = shape + batch_shape
            output = ops.uniform_random(
                output_shape, seed=seed) * (tensor.zeros(
                    output_shape, dtype='float32') +
                                            (self.high - self.low)) + self.low
            if self.all_arg_is_float:
                return nn.reshape(output, shape)
            else:
                return output

    def log_prob(self, value):
        """Log probability density/mass function.
        Args:
          value: `Variable`.
        Returns:
          log_prob: a `Variable`.
        """
        lb_bool = control_flow.less_than(self.low, value)
        ub_bool = control_flow.less_than(value, self.high)
        lb = tensor.cast(lb_bool, dtype='float32')
        ub = tensor.cast(ub_bool, dtype='float32')
        return nn.log(lb * ub) - nn.log(self.high - self.low)

    def entropy(self):
        """Shannon entropy in nats."""
        return nn.log(self.high - self.low)


class Normal(Distribution):
    """The Normal distribution with location `loc` and `scale` parameters.
    #### Mathematical details
    The probability density function (pdf) is,
    ```none
    pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
    Z = (2 pi sigma**2)**0.5
    ```
    where `loc = mu` is the mean, `scale = sigma` is the std. deviation, and, `Z`
    is the normalization constant.
    The Normal distribution is a member of the [location-scale family](
    https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
    constructed as,
    ```none
    X ~ Normal(loc=0, scale=1)
    Y = loc + scale * X
    ```
    #### Examples
    Examples of initialization of one or a batch of distributions.
    ```python
    # Define a single scalar Normal distribution.
    dist = Normal(loc=0., scale=3.)
    # Define a batch of two scalar valued Normals.
    # The first has mean 1 and standard deviation 11, the second 2 and 22.
    dist = Normal(loc=[1, 2.], scale=[11, 22.])
    # Get 3 samples, returning a 3 x 2 tensor.
    dist.sample([3])
    ```
    Arguments are broadcast when possible.
    ```python
    # Define a batch of two scalar valued Normals.
    # Both have mean 1, but different standard deviations.
    dist = Normal(loc=1., scale=[11, 22.])

    ```
    """

    def __init__(self, loc, scale):
        self.batch_size_unknown = False
        self.all_arg_is_float = False
        if self._validate_args(loc, scale):
            self.batch_size_unknown = True
            self.loc = loc
            self.scale = scale
        else:
            if isinstance(loc, float) and isinstance(scale, float):
                self.all_arg_is_float = True
            self.loc, self.scale = self._to_variable(loc, scale)

    def sample(self, shape, seed=0):
        """Generate samples of the specified shape.
        Args:
          shape: 1D `int32`. Shape of the generated samples.
          seed: Python integer seed for RNG
        Returns:
          samples: a `Variable` with prepended dimensions `shape`.
        """
        batch_shape = list((self.loc + self.scale).shape)

        if self.batch_size_unknown:
            output_shape = shape + batch_shape
            zero_tmp = tensor.fill_constant_batch_size_like(
                self.loc + self.scale, batch_shape + shape, 'float32', 0.)
            normal_random_tmp = nn.gaussian_random_batch_size_like(
                zero_tmp, zero_tmp.shape, mean=0., std=1., seed=seed)
            output = normal_random_tmp * (zero_tmp + self.scale) + self.loc
            return nn.reshape(output, output_shape)
        else:
            output_shape = shape + batch_shape
            output = nn.gaussian_random(output_shape, mean=0., std=1., seed=seed) * \
                     (tensor.zeros(output_shape, dtype='float32') + self.scale) + self.loc
            if self.all_arg_is_float:
                return nn.reshape(output, shape)
            else:
                return output

    def entropy(self):
        """Shannon entropy in nats."""
        batch_shape = list((self.loc + self.scale).shape)
        zero_tmp = tensor.fill_constant_batch_size_like(
            self.loc + self.scale, batch_shape, 'float32', 0.)
        return 0.5 + 0.5 * math.log(2 * math.pi) + nn.log(
            (self.scale + zero_tmp))

    def log_prob(self, value):
        """Log probability density/mass function.
        Args:
          value: `Variable`.
        Returns:
          log_prob: a `Variable`.
        """
        var = self.scale * self.scale
        log_scale = nn.log(self.scale)
        return -1. * ((value - self.loc) * (value - self.loc)) / (
            2. * var) - log_scale - math.log(math.sqrt(2. * math.pi))

    def kl_divergence(self, other):
        """The KL-divergence between self normal distributions.
        Args:
            other: instance of Normal.
        Returns:
            kl_divergence: a `Variable`.
        """
        assert isinstance(other, Normal), "another distribution must be Normal"
        var_ratio = self.scale / other.scale
        var_ratio = (var_ratio * var_ratio)
        t1 = (self.loc - other.loc) / other.scale
        t1 = (t1 * t1)
        return 0.5 * (var_ratio + t1 - 1. - nn.log(var_ratio))
