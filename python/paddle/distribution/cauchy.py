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

import numbers

import numpy as np

import paddle
from paddle.base import framework
from paddle.distribution import distribution


class Cauchy(distribution.Distribution):
    r"""Cauchy distribution is also called Cauchy–Lorentz distribution. It is a continuous probability distribution named after Augustin-Louis Cauchy and Hendrik Lorentz. It has a very wide range of applications in natural sciences.

    The Cauchy distribution has the probability density function (PDF):

    .. math::

        { f(x; loc, scale) = \frac{1}{\pi scale \left[1 + \left(\frac{x - loc}{ scale}\right)^2\right]} = { 1 \over \pi } \left[ {  scale \over (x - loc)^2 +  scale^2 } \right], }

    Args:
        loc (float|Tensor): Location of the peak of the distribution. The data type is float32 or float64.
        scale (float|Tensor): The half-width at half-maximum (HWHM). The data type is float32 or float64. Must be positive values.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import Cauchy

            >>> # init Cauchy with float
            >>> rv = Cauchy(loc=0.1, scale=1.2)
            >>> print(rv.entropy())
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    2.71334577)

            >>> # init Cauchy with N-Dim tensor
            >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
            >>> print(rv.entropy())
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [2.53102422, 3.22417140])
    """

    def __init__(self, loc, scale, name=None):
        self.name = name if name is not None else 'Cauchy'

        if not isinstance(loc, (numbers.Real, framework.Variable)):
            raise TypeError(
                f"Expected type of loc is Real|Variable, but got {type(loc)}"
            )
        if not isinstance(scale, (numbers.Real, framework.Variable)):
            raise TypeError(
                f"Expected type of scale is Real|Variable, but got {type(scale)}"
            )

        if isinstance(loc, numbers.Real):
            loc = paddle.full(shape=(), fill_value=loc)

        if isinstance(scale, numbers.Real):
            scale = paddle.full(shape=(), fill_value=scale)

        if loc.shape != scale.shape:
            self.loc, self.scale = paddle.broadcast_tensors([loc, scale])
        else:
            self.loc, self.scale = loc, scale

        self.dtype = self.loc.dtype

        super().__init__(batch_shape=self.loc.shape, event_shape=())

    @property
    def mean(self):
        """Mean of Cauchy distribution."""
        raise ValueError("Cauchy distribution has no mean.")

    @property
    def variance(self):
        """Variance of Cauchy distribution."""
        raise ValueError("Cauchy distribution has no variance.")

    @property
    def stddev(self):
        """Standard Deviation of Cauchy distribution."""
        raise ValueError("Cauchy distribution has no stddev.")

    def sample(self, shape, name=None):
        """Sample from Cauchy distribution.

        Note:
            `sample` method has no grad, if you want so, please use `rsample` instead.

        Args:
            shape (Sequence[int]): Sample shape.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

        Returns:
            Tensor: Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.sample([10]).shape)
                [10]

                >>> # init Cauchy with 0-Dim tensor
                >>> rv = Cauchy(loc=paddle.full((), 0.1), scale=paddle.full((), 1.2))
                >>> print(rv.sample([10]).shape)
                [10]

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.sample([10]).shape)
                [10, 2]

                >>> # sample 2-Dim data
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.sample([10, 2]).shape)
                [10, 2]

                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.sample([10, 2]).shape)
                [10, 2, 2]
        """
        name = name if name is not None else (self.name + '_sample')
        with paddle.no_grad():
            return self.rsample(shape, name)

    def rsample(self, shape, name=None):
        """Sample from Cauchy distribution (reparameterized).

        Args:
            shape (Sequence[int]): Sample shape.
            name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

        Returns:
            Tensor: Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.rsample([10]).shape)
                [10]

                >>> # init Cauchy with 0-Dim tensor
                >>> rv = Cauchy(loc=paddle.full((), 0.1), scale=paddle.full((), 1.2))
                >>> print(rv.rsample([10]).shape)
                [10]

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.rsample([10]).shape)
                [10, 2]

                >>> # sample 2-Dim data
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.rsample([10, 2]).shape)
                [10, 2]

                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.rsample([10, 2]).shape)
                [10, 2, 2]
        """
        name = name if name is not None else (self.name + '_rsample')

        if not isinstance(shape, (np.ndarray, framework.Variable, list, tuple)):
            raise TypeError(
                f"Expected type of shape is Sequence[int], but got {type(shape)}"
            )

        shape = shape if isinstance(shape, tuple) else tuple(shape)
        shape = self._extend_shape(shape)

        loc = self.loc.expand(shape)
        scale = self.scale.expand(shape)
        uniforms = paddle.rand(shape, dtype=self.dtype)
        return paddle.add(
            loc,
            paddle.multiply(scale, paddle.tan(np.pi * (uniforms - 0.5))),
            name=name,
        )

    def prob(self, value):
        r"""Probability density function(PDF) evaluated at value.

        .. math::

            { f(x; loc, scale) = \frac{1}{\pi scale \left[1 + \left(\frac{x - loc}{ scale}\right)^2\right]} = { 1 \over \pi } \left[ {  scale \over (x - loc)^2 +  scale^2 } \right], }

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: PDF evaluated at value.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.prob(paddle.to_tensor(1.5)))
                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                        0.11234467)

                >>> # broadcast to value
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.prob(paddle.to_tensor([1.5, 5.1])))
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [0.11234467, 0.01444674])

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor([0.1, 0.1]), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.prob(paddle.to_tensor([1.5, 5.1])))
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [0.10753712, 0.02195240])

                >>> # init Cauchy with N-Dim tensor with broadcast
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.prob(paddle.to_tensor([1.5, 5.1])))
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [0.10753712, 0.02195240])
        """
        name = self.name + '_prob'

        if not isinstance(value, (framework.Variable, paddle.pir.Value)):
            raise TypeError(
                f"Expected type of value is Variable or Value, but got {type(value)}"
            )

        return self.log_prob(value).exp(name=name)

    def log_prob(self, value):
        """Log of probability density function.

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: Log of probability density evaluated at value.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.log_prob(paddle.to_tensor(1.5)))
                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                        -2.18618369)

                >>> # broadcast to value
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.log_prob(paddle.to_tensor([1.5, 5.1])))
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [-2.18618369, -4.23728657])

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor([0.1, 0.1]), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.log_prob(paddle.to_tensor([1.5, 5.1])))
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [-2.22991920, -3.81887865])

                >>> # init Cauchy with N-Dim tensor with broadcast
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.log_prob(paddle.to_tensor([1.5, 5.1])))
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [-2.22991920, -3.81887865])
        """
        name = self.name + '_log_prob'

        if not isinstance(value, (framework.Variable, paddle.pir.Value)):
            raise TypeError(
                f"Expected type of value is Variable or Value, but got {type(value)}"
            )

        value = self._check_values_dtype_in_probs(self.loc, value)
        loc, scale, value = paddle.broadcast_tensors(
            [self.loc, self.scale, value]
        )

        return paddle.subtract(
            -(
                paddle.square(paddle.divide(paddle.subtract(value, loc), scale))
            ).log1p(),
            paddle.add(
                paddle.full(loc.shape, np.log(np.pi), dtype=self.dtype),
                scale.log(),
            ),
            name=name,
        )

    def cdf(self, value):
        r"""Cumulative distribution function(CDF) evaluated at value.

        .. math::

            { \frac{1}{\pi} \arctan\left(\frac{x-loc}{ scale}\right)+\frac{1}{2}\! }

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: CDF evaluated at value.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.cdf(paddle.to_tensor(1.5)))
                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                        0.77443725)

                >>> # broadcast to value
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.cdf(paddle.to_tensor([1.5, 5.1])))
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [0.77443725, 0.92502367])

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor([0.1, 0.1]), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.cdf(paddle.to_tensor([1.5, 5.1])))
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [0.80256844, 0.87888104])

                >>> # init Cauchy with N-Dim tensor with broadcast
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.cdf(paddle.to_tensor([1.5, 5.1])))
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [0.80256844, 0.87888104])
        """
        name = self.name + '_cdf'

        if not isinstance(value, (framework.Variable, paddle.pir.Value)):
            raise TypeError(
                f"Expected type of value is Variable or Value, but got {type(value)}"
            )

        value = self._check_values_dtype_in_probs(self.loc, value)
        loc, scale, value = paddle.broadcast_tensors(
            [self.loc, self.scale, value]
        )

        return (
            paddle.atan(
                paddle.divide(paddle.subtract(value, loc), scale), name=name
            )
            / np.pi
            + 0.5
        )

    def entropy(self):
        r"""Entropy of Cauchy distribution.

        .. math::

            { \log(4\pi scale)\! }

        Returns:
            Tensor: Entropy of distribution.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> # init Cauchy with float
                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> print(rv.entropy())
                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                        2.71334577)

                >>> # init Cauchy with N-Dim tensor
                >>> rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
                >>> print(rv.entropy())
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [2.53102422, 3.22417140])

        """
        name = self.name + '_entropy'
        return paddle.add(
            paddle.full(self.loc.shape, np.log(4 * np.pi), dtype=self.dtype),
            self.scale.log(),
            name=name,
        )

    def kl_divergence(self, other):
        """The KL-divergence between two Cauchy distributions.

        Note:
            [1] Frédéric Chyzak, Frank Nielsen, A closed-form formula for the Kullback-Leibler divergence between Cauchy distributions, 2019

        Args:
            other (Cauchy): instance of Cauchy.

        Returns:
            Tensor: kl-divergence between two Cauchy distributions.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Cauchy

                >>> rv = Cauchy(loc=0.1, scale=1.2)
                >>> rv_other = Cauchy(loc=paddle.to_tensor(1.2), scale=paddle.to_tensor([2.3, 3.4]))
                >>> print(rv.kl_divergence(rv_other))
                Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [0.19819736, 0.31532931])
        """
        name = self.name + '_kl_divergence'

        if not isinstance(other, Cauchy):
            raise TypeError(
                f"Expected type of other is Cauchy, but got {type(other)}"
            )

        a_loc = self.loc
        b_loc = other.loc

        a_scale = self.scale
        b_scale = other.scale

        t1 = paddle.add(
            paddle.pow(paddle.add(a_scale, b_scale), 2),
            paddle.pow(paddle.subtract(a_loc, b_loc), 2),
        ).log()
        t2 = (4 * paddle.multiply(a_scale, b_scale)).log()

        return paddle.subtract(t1, t2, name=name)
