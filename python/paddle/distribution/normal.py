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

import numpy as np
from paddle import _C_ops
from paddle.distribution import distribution
from paddle.fluid import core
from paddle.fluid.data_feeder import (check_dtype, check_type,
                                      check_variable_and_dtype, convert_dtype)
from paddle.fluid.framework import _non_static_mode, in_dygraph_mode
from paddle.fluid.layers import (control_flow, elementwise_add, elementwise_div,
                                 elementwise_mul, elementwise_sub, nn, ops,
                                 tensor)


class Normal(distribution.Distribution):
    r"""The Normal distribution with location `loc` and `scale` parameters.

    Mathematical details

    The probability density function (pdf) is

    .. math::

        pdf(x; \mu, \sigma) = \\frac{1}{Z}e^{\\frac {-0.5 (x - \mu)^2}  {\sigma^2} }

    .. math::

        Z = (2 \pi \sigma^2)^{0.5}

    In the above equation:

    * :math:`loc = \mu`: is the mean.
    * :math:`scale = \sigma`: is the std.
    * :math:`Z`: is the normalization constant.

    Args:
        loc(int|float|list|tuple|numpy.ndarray|Tensor): The mean of normal distribution.The data type is int, float, list, numpy.ndarray or Tensor.
        scale(int|float|list|tuple|numpy.ndarray|Tensor): The std of normal distribution.The data type is int, float, list, numpy.ndarray or Tensor.
        name(str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python
          
          import paddle
          from paddle.distribution import Normal

          # Define a single scalar Normal distribution.
          dist = Normal(loc=0., scale=3.)
          # Define a batch of two scalar valued Normals.
          # The first has mean 1 and standard deviation 11, the second 2 and 22.
          dist = Normal(loc=[1., 2.], scale=[11., 22.])
          # Get 3 samples, returning a 3 x 2 tensor.
          dist.sample([3])

          # Define a batch of two scalar valued Normals.
          # Both have mean 1, but different standard deviations.
          dist = Normal(loc=1., scale=[11., 22.])

          # Complete example
          value_tensor = paddle.to_tensor([0.8], dtype="float32")

          normal_a = Normal([0.], [1.])
          normal_b = Normal([0.5], [2.])
          sample = normal_a.sample([2])
          # a random tensor created by normal distribution with shape: [2, 1]
          entropy = normal_a.entropy()
          # [1.4189385] with shape: [1]
          lp = normal_a.log_prob(value_tensor)
          # [-1.2389386] with shape: [1]
          p = normal_a.probs(value_tensor)
          # [0.28969154] with shape: [1]
          kl = normal_a.kl_divergence(normal_b)
          # [0.34939718] with shape: [1]
    """

    def __init__(self, loc, scale, name=None):
        if not _non_static_mode():
            check_type(loc, 'loc',
                       (int, float, np.ndarray, tensor.Variable, list, tuple),
                       'Normal')
            check_type(scale, 'scale',
                       (int, float, np.ndarray, tensor.Variable, list, tuple),
                       'Normal')

        self.batch_size_unknown = False
        self.all_arg_is_float = False
        self.name = name if name is not None else 'Normal'
        self.dtype = 'float32'

        if isinstance(loc, int):
            loc = float(loc)
        if isinstance(scale, int):
            scale = float(scale)

        if self._validate_args(loc, scale):
            self.batch_size_unknown = True
            self.loc = loc
            self.scale = scale
            self.dtype = convert_dtype(loc.dtype)
        else:
            if isinstance(loc, float) and isinstance(scale, float):
                self.all_arg_is_float = True
            if isinstance(
                    loc,
                    np.ndarray) and str(loc.dtype) in ['float32', 'float64']:
                self.dtype = loc.dtype
            elif isinstance(
                    scale,
                    np.ndarray) and str(scale.dtype) in ['float32', 'float64']:
                self.dtype = scale.dtype
            # pylint: disable=unbalanced-tuple-unpacking
            self.loc, self.scale = self._to_tensor(loc, scale)
            if self.dtype != convert_dtype(self.loc.dtype):
                self.loc = tensor.cast(self.loc, dtype=self.dtype)
                self.scale = tensor.cast(self.scale, dtype=self.dtype)
        super(Normal, self).__init__(self.loc.shape)

    def sample(self, shape, seed=0):
        """Generate samples of the specified shape.

        Args:
          shape (list): 1D `int32`. Shape of the generated samples.
          seed (int): Python integer number.

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.

        """
        if not _non_static_mode():
            check_type(shape, 'shape', (list), 'sample')
            check_type(seed, 'seed', (int), 'sample')

        batch_shape = list((self.loc + self.scale).shape)
        name = self.name + '_sample'

        if self.batch_size_unknown:
            output_shape = shape + batch_shape
            zero_tmp = tensor.fill_constant_batch_size_like(
                self.loc + self.scale, batch_shape + shape, self.dtype, 0.)
            zero_tmp_reshape = nn.reshape(zero_tmp, output_shape)
            zero_tmp_shape = nn.shape(zero_tmp_reshape)
            normal_random_tmp = nn.gaussian_random(
                zero_tmp_shape, mean=0., std=1., seed=seed, dtype=self.dtype)
            output = normal_random_tmp * (zero_tmp_reshape + self.scale)
            output = elementwise_add(output, self.loc, name=name)
            return output
        else:
            output_shape = shape + batch_shape
            output = nn.gaussian_random(output_shape, mean=0., std=1., seed=seed, dtype=self.dtype) * \
                     (tensor.zeros(output_shape, dtype=self.dtype) + self.scale)
            output = elementwise_add(output, self.loc, name=name)
            if self.all_arg_is_float:
                return nn.reshape(output, shape, name=name)
            else:
                return output

    def entropy(self):
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            entropy(\sigma) = 0.5 \\log (2 \pi e \sigma^2)

        In the above equation:

        * :math:`scale = \sigma`: is the std.

        Returns:
          Tensor: Shannon entropy of normal distribution.The data type is float32.

        """
        name = self.name + '_entropy'
        batch_shape = list((self.loc + self.scale).shape)
        zero_tmp = tensor.fill_constant_batch_size_like(
            self.loc + self.scale, batch_shape, self.dtype, 0.)
        return elementwise_add(
            0.5 + zero_tmp,
            0.5 * math.log(2 * math.pi) + nn.log((self.scale + zero_tmp)),
            name=name)

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability.The data type is same with value.

        """
        name = self.name + '_log_prob'
        value = self._check_values_dtype_in_probs(self.loc, value)

        var = self.scale * self.scale
        log_scale = nn.log(self.scale)
        return elementwise_sub(
            -1. * ((value - self.loc) * (value - self.loc)) / (2. * var),
            log_scale + math.log(math.sqrt(2. * math.pi)),
            name=name)

    def probs(self, value):
        """Probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: probability.The data type is same with value.

        """
        name = self.name + '_probs'
        value = self._check_values_dtype_in_probs(self.loc, value)

        var = self.scale * self.scale
        return elementwise_div(
            ops.exp(-1. * ((value - self.loc) * (value - self.loc)) /
                    (2. * var)), (math.sqrt(2 * math.pi) * self.scale),
            name=name)

    def kl_divergence(self, other):
        r"""The KL-divergence between two normal distributions.

        The probability density function (pdf) is

        .. math::

            KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = 0.5 (ratio^2 + (\\frac{diff}{\sigma_1})^2 - 1 - 2 \\ln {ratio})

        .. math::

            ratio = \\frac{\sigma_0}{\sigma_1}
        
        .. math::

            diff = \mu_1 - \mu_0

        In the above equation:

        * :math:`loc = \mu_0`: is the mean of current Normal distribution.
        * :math:`scale = \sigma_0`: is the std of current Normal distribution.
        * :math:`loc = \mu_1`: is the mean of other Normal distribution.
        * :math:`scale = \sigma_1`: is the std of other Normal distribution.
        * :math:`ratio`: is the ratio of scales.
        * :math:`diff`: is the difference between means.

        Args:
            other (Normal): instance of Normal.

        Returns:
            Tensor: kl-divergence between two normal distributions.The data type is float32.

        """
        if not _non_static_mode():
            check_type(other, 'other', Normal, 'kl_divergence')

        name = self.name + '_kl_divergence'
        var_ratio = self.scale / other.scale
        var_ratio = (var_ratio * var_ratio)
        t1 = (self.loc - other.loc) / other.scale
        t1 = (t1 * t1)
        return elementwise_add(
            0.5 * var_ratio, 0.5 * (t1 - 1. - nn.log(var_ratio)), name=name)
