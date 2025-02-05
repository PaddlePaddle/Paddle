# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import paddle
from paddle.base.data_feeder import check_type
from paddle.base.framework import Variable
from paddle.distribution import Gamma, distribution
from paddle.framework import in_dynamic_mode

if TYPE_CHECKING:
    from paddle import Tensor, dtype


class StudentT(distribution.Distribution):
    r"""
    The StudentT distribution with parameters: `df`, `loc`, `scale`.

    In probability theory and statistics, the StudentT distribution is one of the basic continuous probability distributions
    defined on the real number set.

    The probability density function (pdf) is

    .. math::

        pdf(x; \nu, \mu, \sigma) = \frac{\Gamma[(\nu+1)/2]}{\sigma\sqrt{\nu\pi}\Gamma(\nu/2)[1+(\frac{x-\mu}{\sigma})^2/\nu]^{(1+\nu)/2}}

    In the above equation:

    * :math:`df = \nu`: is the degree of freedom.
    * :math:`loc = \mu`: is the center parameter.
    * :math:`scale = \sigma`: is the scale parameter.
    * :math:`\Gamma(\cdot)`: is the gamma function.

    Args:
        df (float|Tensor): The degree of freedom of the distribution, which should be non-negative. If the input data type is float,
            the data type of `df` will be converted to a 1-D Tensor with paddle global default dtype. Supported dtype: float32, float64.
        loc (float|Tensor): The center of the distribution. If the input data type is float, the data type of `loc` will be converted to a
            1-D Tensor with paddle global default dtype. Supported dtype: float32, float64.
        scale (float|Tensor): The scale of the distribution, which should be non-negative. If the input data type is float, the data type
            of `scale` will be converted to a 1-D Tensor with paddle global default dtype. Supported dtype: float32, float64.
        name(str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import StudentT
            >>> paddle.set_device('cpu')
            >>> paddle.seed(100)
            >>> dist = StudentT(df=10.0, loc=0.0, scale=1.0)
            >>> dist.sample([3])
            Tensor(shape=[3, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[-2.07709980],
             [ 0.27981189],
             [ 0.00881413]])

            >>> dist2 = StudentT(df=paddle.to_tensor([10.0, 5.0]), loc=paddle.to_tensor([0.0, 0.0]), scale=paddle.to_tensor([1.0, 2.0]))
            >>> value_tensor = paddle.to_tensor([0.8], dtype="float32")
            >>> lp = dist2.log_prob(value_tensor)
            >>> print(lp)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-1.28509235, -1.75626254])

            >>> p = dist2.prob(value_tensor)
            >>> print(p)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [0.27662504, 0.17268908])

            >>> entropy = dist2.entropy()
            >>> print(entropy)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.52126312, 2.32064891])

    """

    df: Tensor
    loc: Tensor
    scale: Tensor
    name: str
    dtype: dtype

    def __init__(
        self,
        df: float | Tensor,
        loc: float | Tensor,
        scale: float | Tensor,
        name: str | None = None,
    ) -> None:
        if not in_dynamic_mode():
            check_type(
                df,
                'df',
                (
                    float,
                    Variable,
                    paddle.pir.Value,
                ),
                'StudentT',
            )
            check_type(
                loc,
                'loc',
                (
                    float,
                    Variable,
                    paddle.pir.Value,
                ),
                'StudentT',
            )
            check_type(
                scale,
                'scale',
                (
                    float,
                    Variable,
                    paddle.pir.Value,
                ),
                'StudentT',
            )

        self.name = name if name is not None else 'StudentT'
        self.df, self.loc, self.scale = self._broadcast_all(df, loc, scale)

        if not self._check_nonnegative(self.df):
            raise ValueError(
                'Every element of input parameter `df` should be nonnegative.'
            )
        if not self._check_nonnegative(self.scale):
            raise ValueError(
                'Every element of input parameter `scale` should be nonnegative.'
            )

        batch_shape = self.df.shape
        super().__init__(batch_shape)
        self._chi2 = Gamma(0.5 * self.df, paddle.full_like(self.df, 0.5))

    def _check_nonnegative(self, value: Tensor) -> bool:
        """Check the non-negative constraint for input parameters

        Args:
            value (Tensor)

        Returns:
            bool: pass or not.
        """
        return (value >= 0.0).all()

    @property
    def mean(self) -> Tensor:
        """Mean of StudentT distribution.

        Returns:
            Tensor: mean value.
        """
        return paddle.where(
            self.df > 1.0,
            self.loc,
            paddle.full_like(self.loc, fill_value=float('nan')),
        )

    @property
    def variance(self) -> Tensor:
        """Variance of StudentT distribution.

        Returns:
            Tensor: variance value.
        """
        var = self.df.clone().detach()
        var_condition = self.df > 2.0
        var = paddle.where(
            var_condition,
            self.scale.pow(2) * var / (var - 2),
            paddle.full_like(var, fill_value=float('nan')),
        )
        inf_condition = (self.df <= 2.0).logical_and(self.df > 1.0)
        var = paddle.where(
            inf_condition, paddle.full_like(var, fill_value=float('inf')), var
        )
        return var

    def sample(self, shape: Sequence[int] = []) -> Tensor:
        """Generate StudentT samples of the specified shape. The final shape would be ``shape+batch_shape`` .

        Args:
            shape (Sequence[int], optional): Prepended shape of the generated samples.

        Returns:
            Tensor: Sampled data with shape `sample_shape` + `batch_shape`.
        """
        if not isinstance(shape, Sequence):
            raise TypeError('sample shape must be Sequence object.')

        output_shape = self._extend_shape(shape)
        z = paddle.normal(shape=output_shape)
        chi2 = self._chi2.sample(shape)
        x = z * paddle.rsqrt(chi2 / self.df)
        return self.loc + self.scale * x

    def entropy(self) -> Tensor:
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            H = \log(\frac{\Gamma(\nu/2)\Gamma(1/2) \sigma \sqrt{\nu}}{\Gamma[(1+\nu)/2]}) + \frac{(1+\nu)}{2} \cdot \{\psi[(1+\nu)/2] - \psi(\nu/2)\}

        In the above equation:

        * :math:`\nu`: is the degree of freedom.
        * :math:`\Gamma()`: is the gamma function.
        * :math:`\psi()`: is the digamma function.

        Returns:
            Tensor: Shannon entropy of StudentT distribution. The data type is the same as `df`.
        """
        lbeta = (
            paddle.lgamma(0.5 * self.df)
            + math.lgamma(0.5)
            - paddle.lgamma(0.5 * (self.df + 1))
        )
        return (
            self.scale.log()
            + 0.5
            * (self.df + 1)
            * (
                paddle.digamma(0.5 * (self.df + 1))
                - paddle.digamma(0.5 * self.df)
            )
            + 0.5 * self.df.log()
            + lbeta
        )

    def log_prob(self, value: Tensor) -> Tensor:
        """Log probability density function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability density. The data type is the same as `df`.
        """
        value = self._check_values_dtype_in_probs(self.df, value)
        y = (value - self.loc) / self.scale
        Z = (
            self.scale.log()
            + 0.5 * self.df.log()
            + 0.5 * math.log(math.pi)
            + paddle.lgamma(0.5 * self.df)
            - paddle.lgamma(0.5 * (self.df + 1.0))
        )
        return -0.5 * (self.df + 1.0) * paddle.log1p(y**2.0 / self.df) - Z

    def prob(self, value: Tensor) -> Tensor:
        """Probability density function.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: probability density. The data type is the same as `df`.
        """
        return paddle.exp(self.log_prob(value))
