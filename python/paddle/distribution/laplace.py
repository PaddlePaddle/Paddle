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

import numpy as np
import paddle
from paddle.distribution import distribution
from paddle.fluid.data_feeder import (check_dtype, check_type,
                                      check_variable_and_dtype, convert_dtype)
from paddle.fluid.framework import _non_static_mode, in_dygraph_mode
from paddle.fluid.layers import (elementwise_add, elementwise_div,
                                 elementwise_mul, elementwise_sub, nn, ops,
                                 tensor)


class Laplace(distribution.Distribution):

    def __init__(self, loc, scale, name=None):
        if not _non_static_mode():
            check_type(loc, 'loc',
                       (int, float, np.ndarray, tensor.Variable, list, tuple),
                       'Laplace')
            check_type(scale, 'scale',
                       (int, float, np.ndarray, tensor.Variable, list, tuple),
                       'Laplace')

        self.batch_size_unknown = False
        self.all_arg_is_float = False
        self.name = name if name is not None else 'Laplace'
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
            if isinstance(loc, np.ndarray) and str(
                    loc.dtype) in ['float32', 'float64']:
                self.dtype = loc.dtype
            elif isinstance(scale, np.ndarray) and str(
                    scale.dtype) in ['float32', 'float64']:
                self.dtype = scale.dtype
            # pylint: disable=unbalanced-tuple-unpacking
            self.loc, self.scale = self._to_tensor(loc, scale)
            if self.dtype != convert_dtype(self.loc.dtype):
                self.loc = tensor.cast(self.loc, dtype=self.dtype)
                self.scale = tensor.cast(self.scale, dtype=self.dtype)
        super(Laplace, self).__init__(self.loc.shape)

    @property
    def mean(self):
        """Mean of distribution"""
        return self.loc

    @property
    def stddev(self):
        """standard deviation"""
        return (2**0.5) * self.scale

    @property
    def variance(self):
        """Variance of distribution"""
        return self.stddev.pow(2)

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability.The data type is same with value.

        """
        name = self.name + '_log_prob'
        value = self._check_values_dtype_in_probs(self.loc, value)
        log_scale = -nn.log(2 * self.scale)

        return elementwise_sub(log_scale,\
                               ops.abs(value - self.loc) / self.scale,\
                               name=name)

    def probs(self, value):
        """Probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: probability.The data type is same with value.

        """
        return self.prob(value)

    def entropy(self):
        """Entropy of Laplace distribution.

        Returns:
            Entropy of distribution.
        """
        return 1 + nn.log(2 * self.scale)

    def cdf(self, value):
        """Cumulative distribution function
        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: cumulative probability of value.
        """
        name = self.name + '_cdf'
        value = self._check_values_dtype_in_probs(self.loc, value)
        iterm = 0.5 * (value - self.loc).sign() \
            * paddle.expm1(-(value - self.loc).abs() / self.scale)
        return elementwise_add(0.5, iterm, name=name)

    def icdf(self, value):
        """Independant Cumulative distribution function
        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: cumulative probability of value.
        """
        name = self.name + '_icdf'
        term = value - 0.5
        return elementwise_sub(self.loc, \
                    self.scale * (term).sign() * paddle.log1p(-2 * term.abs()),
                    name=name)

    def sample(self, shape=()):
        """Generate samples of the specified shape.

        Args:
          shape (list): 1D `int32`. Shape of the generated samples.
          seed (int): Python integer number.

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.

        """
        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape, seed=0):
        """reparameterized sample
        Args:
          shape (list): 1D `int32`. Shape of the generated samples.
          seed (int): Python integer number.

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.
        """

        try:
            from paddle import finfo
            finfo = finfo(self.loc.dtype)
            eps = finfo.eps
        except:
            if self.loc.dtype == paddle.float16 or\
               self.loc.dtype == paddle.float32 or\
               self.loc.dtype == paddle.complex64:
                eps = 1.19209e-07
            elif self.loc.dtype == paddle.float64 or\
                 self.loc.dtype == paddle.complex128:
                eps = 2.22045e-16
            else:
                raise TypeError("self.loc requires a floating point type")

        if not _non_static_mode():
            check_type(shape, 'shape', (list), 'rsample')
            check_type(seed, 'seed', (int), 'rsample')

        name = self.name + '_rsample'
        shape = self._extend_shape(shape)
        u = paddle.uniform(shape=shape, min=eps - 1, max=1)

        return elementwise_sub(self.loc, \
            self.scale * u.sign() * paddle.log1p(-u.abs()), \
            name=name)
