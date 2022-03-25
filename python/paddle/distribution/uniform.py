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

from ..fluid import core
from ..fluid.data_feeder import (check_dtype, check_type,
                                 check_variable_and_dtype, convert_dtype)
from ..fluid.framework import _non_static_mode
from ..fluid.layers import (control_flow, elementwise_add, elementwise_div,
                            elementwise_mul, elementwise_sub, nn, ops, tensor)
from ..tensor import arange, concat, gather_nd, multinomial
from .distribution import Distribution


class Uniform(Distribution):
    r"""Uniform distribution with `low` and `high` parameters.

    Mathematical Details

    The probability density function (pdf) is

    .. math::

        pdf(x; a, b) = \\frac{1}{Z}, \ a <=x <b

    .. math::

        Z = b - a

    In the above equation:

    * :math:`low = a`,
    * :math:`high = b`,
    * :math:`Z`: is the normalizing constant.

    The parameters `low` and `high` must be shaped in a way that supports
    [broadcasting](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/beginners_guide/basic_concept/broadcasting_en.html) (e.g., `high - low` is a valid operation).

    Args:
        low(int|float|list|tuple|numpy.ndarray|Tensor): The lower boundary of uniform distribution.The data type is int, float, list, numpy.ndarray or Tensor
        high(int|float|list|tuple|numpy.ndarray|Tensor): The higher boundary of uniform distribution.The data type is int, float, list, numpy.ndarray or Tensor
        name(str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

          import paddle
          from paddle.distribution import Uniform

          # Without broadcasting, a single uniform distribution [3, 4]:
          u1 = Uniform(low=3.0, high=4.0)
          # 2 distributions [1, 3], [2, 4]
          u2 = Uniform(low=[1.0, 2.0], high=[3.0, 4.0])
          # 4 distributions
          u3 = Uniform(low=[[1.0, 2.0], [3.0, 4.0]],
                    high=[[1.5, 2.5], [3.5, 4.5]])

          # With broadcasting:
          u4 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])

          # Complete example
          value_tensor = paddle.to_tensor([0.8], dtype="float32")

          uniform = Uniform([0.], [2.])

          sample = uniform.sample([2])
          # a random tensor created by uniform distribution with shape: [2, 1]
          entropy = uniform.entropy()
          # [0.6931472] with shape: [1]
          lp = uniform.log_prob(value_tensor)
          # [-0.6931472] with shape: [1]
          p = uniform.probs(value_tensor)
          # [0.5] with shape: [1]
    """

    def __init__(self, low, high, name=None):
        if not _non_static_mode():
            check_type(low, 'low',
                       (int, float, np.ndarray, tensor.Variable, list, tuple),
                       'Uniform')
            check_type(high, 'high',
                       (int, float, np.ndarray, tensor.Variable, list, tuple),
                       'Uniform')

        self.all_arg_is_float = False
        self.batch_size_unknown = False
        self.name = name if name is not None else 'Uniform'
        self.dtype = 'float32'

        if isinstance(low, int):
            low = float(low)
        if isinstance(high, int):
            high = float(high)

        if self._validate_args(low, high):
            self.batch_size_unknown = True
            self.low = low
            self.high = high
            self.dtype = convert_dtype(low.dtype)
        else:
            if isinstance(low, float) and isinstance(high, float):
                self.all_arg_is_float = True
            if isinstance(
                    low,
                    np.ndarray) and str(low.dtype) in ['float32', 'float64']:
                self.dtype = low.dtype
            elif isinstance(
                    high,
                    np.ndarray) and str(high.dtype) in ['float32', 'float64']:
                self.dtype = high.dtype
            # pylint: disable=unbalanced-tuple-unpacking
            self.low, self.high = self._to_tensor(low, high)
            if self.dtype != convert_dtype(self.low.dtype):
                self.low = tensor.cast(self.low, dtype=self.dtype)
                self.high = tensor.cast(self.high, dtype=self.dtype)

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

        name = self.name + '_sample'
        batch_shape = list((self.low + self.high).shape)
        if self.batch_size_unknown:
            output_shape = shape + batch_shape
            zero_tmp = tensor.fill_constant_batch_size_like(
                self.low + self.high, batch_shape + shape, self.dtype, 0.)
            uniform_random_tmp = nn.uniform_random_batch_size_like(
                zero_tmp,
                zero_tmp.shape,
                dtype=self.dtype,
                min=0.,
                max=1.,
                seed=seed)
            zero_tmp_reshape = nn.reshape(zero_tmp, output_shape)
            uniform_random_tmp_reshape = nn.reshape(uniform_random_tmp,
                                                    output_shape)
            output = uniform_random_tmp_reshape * (
                zero_tmp_reshape + self.high - self.low)
            output = elementwise_add(output, self.low, name=name)
            return output
        else:
            output_shape = shape + batch_shape
            output = nn.uniform_random(
                output_shape, dtype=self.dtype, min=0., max=1.,
                seed=seed) * (tensor.zeros(
                    output_shape, dtype=self.dtype) + (self.high - self.low))
            output = elementwise_add(output, self.low, name=name)
            if self.all_arg_is_float:
                return nn.reshape(output, shape, name=name)
            else:
                return output

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability.The data type is same with value.

        """
        value = self._check_values_dtype_in_probs(self.low, value)
        if _non_static_mode():
            # ensure value in [low, high]
            lb_bool = self.low < value
            ub_bool = value < self.high

            lb = _C_ops.cast(lb_bool, 'in_dtype', lb_bool.dtype, 'out_dtype',
                             value.dtype)
            ub = _C_ops.cast(ub_bool, 'in_dtype', ub_bool.dtype, 'out_dtype',
                             value.dtype)
            return nn.log(lb * ub) - nn.log(self.high - self.low)

        name = self.name + '_log_prob'
        lb_bool = self.low < value
        ub_bool = value < self.high
        lb = tensor.cast(lb_bool, dtype=value.dtype)
        ub = tensor.cast(ub_bool, dtype=value.dtype)
        return elementwise_sub(
            nn.log(lb * ub), nn.log(self.high - self.low), name=name)

    def probs(self, value):
        """Probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: probability.The data type is same with value.

        """
        value = self._check_values_dtype_in_probs(self.low, value)
        if _non_static_mode():
            lb_bool = self.low < value
            ub_bool = value < self.high

            lb = _C_ops.cast(lb_bool, 'in_dtype', lb_bool.dtype, 'out_dtype',
                             value.dtype)
            ub = _C_ops.cast(ub_bool, 'in_dtype', ub_bool.dtype, 'out_dtype',
                             value.dtype)
            return (lb * ub) / (self.high - self.low)

        name = self.name + '_probs'
        lb_bool = self.low < value
        ub_bool = value < self.high
        lb = tensor.cast(lb_bool, dtype=value.dtype)
        ub = tensor.cast(ub_bool, dtype=value.dtype)
        return elementwise_div((lb * ub), (self.high - self.low), name=name)

    def entropy(self):
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            entropy(low, high) = \\log (high - low)

        Returns:
          Tensor: Shannon entropy of uniform distribution.The data type is float32.

        """
        name = self.name + '_entropy'
        return nn.log(self.high - self.low, name=name)
