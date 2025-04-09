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
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import paddle
from paddle import _C_ops
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.framework import Variable
from paddle.distribution import distribution
from paddle.framework import in_dynamic_mode
from paddle.tensor import random

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Union

    from typing_extensions import TypeAlias

    from paddle import Tensor
    from paddle._typing import NestedSequence

    _UniformBoundary: TypeAlias = Union[
        float,
        Sequence[float],
        NestedSequence[float],
        npt.NDArray[Union[np.float32, np.float64]],
        Tensor,
    ]


class Uniform(distribution.Distribution):
    r"""Uniform distribution with `low` and `high` parameters.

    Mathematical Details

    The probability density function (pdf) is

    .. math::

        pdf(x; a, b) = \frac{1}{Z}, \ a <=x <b

    .. math::

        Z = b - a

    In the above equation:

    * :math:`low = a`,
    * :math:`high = b`,
    * :math:`Z`: is the normalizing constant.

    The parameters `low` and `high` must be shaped in a way that supports
    `Broadcasting` (e.g., `high - low` is a valid operation).

    Note:
        If you want know more about broadcasting, please refer to `Introduction to Tensor`_ .

        .. _Introduction to Tensor: ../../guides/beginner/tensor_en.html#chapter5-broadcasting-of-tensor

    Args:
        low(int|float|list|tuple|numpy.ndarray|Tensor): The lower boundary of
            uniform distribution.The data type is float32 and float64.
        high(int|float|list|tuple|numpy.ndarray|Tensor): The higher boundary
            of uniform distribution.The data type is float32 and float64.
        name (str, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import Uniform
            >>> paddle.seed(2023)

            >>> # Without broadcasting, a single uniform distribution [3, 4]:
            >>> u1 = Uniform(low=3.0, high=4.0)
            >>> # 2 distributions [1, 3], [2, 4]
            >>> u2 = Uniform(low=[1.0, 2.0], high=[3.0, 4.0])
            >>> # 4 distributions
            >>> u3 = Uniform(low=[[1.0, 2.0], [3.0, 4.0]],
            ...             high=[[1.5, 2.5], [3.5, 4.5]])
            ...
            >>> # With broadcasting:
            >>> u4 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])

            >>> # Complete example
            >>> value_tensor = paddle.to_tensor([0.8], dtype="float32")

            >>> uniform = Uniform([0.], [2.])

            >>> sample = uniform.sample([2])
            >>> # a random tensor created by uniform distribution with shape: [2, 1]
            >>> entropy = uniform.entropy()
            >>> print(entropy)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.69314718])

            >>> lp = uniform.log_prob(value_tensor)
            >>> print(lp)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [-0.69314718])

            >>> p = uniform.probs(value_tensor)
            >>> print(p)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.50000000])
    """

    low: Tensor
    high: Tensor

    def __init__(
        self,
        low: _UniformBoundary,
        high: _UniformBoundary,
        name: str | None = None,
    ) -> None:
        if not in_dynamic_mode():
            check_type(
                low,
                'low',
                (
                    int,
                    float,
                    np.ndarray,
                    Variable,
                    paddle.pir.Value,
                    list,
                    tuple,
                ),
                'Uniform',
            )
            check_type(
                high,
                'high',
                (
                    int,
                    float,
                    np.ndarray,
                    Variable,
                    paddle.pir.Value,
                    list,
                    tuple,
                ),
                'Uniform',
            )

        self.all_arg_is_float = False
        self.batch_size_unknown = False
        self.name = name if name is not None else 'Uniform'
        self.dtype = 'float32'

        if isinstance(low, int):
            low = float(low)
        if isinstance(high, int):
            high = float(high)

        if self._validate_args(low, high):
            self.low = low
            self.high = high
            self.dtype = convert_dtype(low.dtype)
        else:
            if isinstance(low, float) and isinstance(high, float):
                self.all_arg_is_float = True
            if isinstance(low, np.ndarray) and str(low.dtype) in [
                'float32',
                'float64',
            ]:
                self.dtype = low.dtype
            elif isinstance(high, np.ndarray) and str(high.dtype) in [
                'float32',
                'float64',
            ]:
                self.dtype = high.dtype
            self.low, self.high = self._to_tensor(low, high)
            if self.dtype != convert_dtype(self.low.dtype):
                self.low = paddle.cast(self.low, dtype=self.dtype)
                self.high = paddle.cast(self.high, dtype=self.dtype)

        super().__init__(self.low.shape)

    def sample(self, shape: Sequence[int] = [], seed: int = 0) -> Tensor:
        """Generate samples of the specified shape.

        Args:
            shape (Sequence[int], optional): 1D `int32`. Shape of the generated samples.
                Defaults to [].
            seed (int): Python integer number.

        Returns:
            Tensor, A tensor with prepended dimensions shape. The data type is float32.

        """
        if not in_dynamic_mode():
            check_type(shape, 'shape', (list, tuple), 'sample')
            check_type(seed, 'seed', (int), 'sample')
        shape = list(shape)
        name = self.name + '_sample'
        batch_shape = list((self.low + self.high).shape)
        if -1 in batch_shape:
            output_shape = shape + batch_shape
            fill_shape = list(batch_shape + shape)
            fill_shape[0] = paddle.shape(self.low + self.high)[0].item()
            zero_tmp = paddle.full(fill_shape, 0.0, self.dtype)
            uniform_random_tmp = random.uniform_random_batch_size_like(
                zero_tmp,
                zero_tmp.shape,
                dtype=self.dtype,
                min=0.0,
                max=1.0,
                seed=seed,
            )
            zero_tmp_reshape = paddle.reshape(zero_tmp, output_shape)
            uniform_random_tmp_reshape = paddle.reshape(
                uniform_random_tmp, output_shape
            )
            output = uniform_random_tmp_reshape * (
                zero_tmp_reshape + self.high - self.low
            )
            output = paddle.add(output, self.low, name=name)
            return output
        else:
            output_shape = shape + batch_shape
            output = paddle.uniform(
                output_shape, dtype=self.dtype, min=0.0, max=1.0, seed=seed
            ) * (
                paddle.zeros(output_shape, dtype=self.dtype)
                + (self.high - self.low)
            )
            output = paddle.add(output, self.low, name=name)
            if self.all_arg_is_float:
                return paddle.reshape(output, shape, name=name)
            else:
                return output

    def log_prob(self, value: Tensor) -> Tensor:
        """Log probability density/mass function.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor, log probability.The data type is same with value.

        """
        value = self._check_values_dtype_in_probs(self.low, value)
        if in_dynamic_mode():
            # ensure value in [low, high]
            lb_bool = self.low < value
            ub_bool = value < self.high

            lb = _C_ops.cast(lb_bool, value.dtype)
            ub = _C_ops.cast(ub_bool, value.dtype)
            return paddle.log(lb * ub) - paddle.log(self.high - self.low)
        else:
            name = self.name + '_log_prob'
            lb_bool = self.low < value
            ub_bool = value < self.high
            lb = paddle.cast(lb_bool, dtype=value.dtype)
            ub = paddle.cast(ub_bool, dtype=value.dtype)
            return paddle.subtract(
                paddle.log(lb * ub), paddle.log(self.high - self.low), name=name
            )

    def probs(self, value: Tensor) -> Tensor:
        """Probability density/mass function.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor, probability. The data type is same with value.

        """
        value = self._check_values_dtype_in_probs(self.low, value)
        if in_dynamic_mode():
            lb_bool = self.low < value
            ub_bool = value < self.high
            lb = _C_ops.cast(lb_bool, value.dtype)
            ub = _C_ops.cast(ub_bool, value.dtype)
            return (lb * ub) / (self.high - self.low)
        else:
            name = self.name + '_probs'
            lb_bool = self.low < value
            ub_bool = value < self.high
            lb = paddle.cast(lb_bool, dtype=value.dtype)
            ub = paddle.cast(ub_bool, dtype=value.dtype)
            return paddle.divide((lb * ub), (self.high - self.low), name=name)

    def entropy(self) -> Tensor:
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            entropy(low, high) = \\log (high - low)

        Returns:
            Tensor, Shannon entropy of uniform distribution.The data type is float32.

        """
        name = self.name + '_entropy'
        return paddle.log(self.high - self.low, name=name)
