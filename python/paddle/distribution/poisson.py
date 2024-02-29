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

from collections.abc import Sequence

import paddle
from paddle.distribution import distribution


class Poisson(distribution.Distribution):
    r"""
    The Poisson distribution with occurrence rate parameter: `rate`.

    In probability theory and statistics, the Poisson distribution is the most basic discrete probability
    distribution defined on the nonnegative integer set, which is used to describe the probability distribution of the number of random
    events occurring per unit time.

    The probability mass function (pmf) is

    .. math::

        pmf(x; \lambda) = \frac{e^{-\lambda} \cdot \lambda^x}{x!}

    In the above equation:

    * :math:`rate = \lambda`: is the mean occurrence rate.

    Args:
        rate(int|float|Tensor): The mean occurrence rate of Poisson distribution which should be greater than 0, meaning the expected occurrence
            times of an event in a fixed time interval. If the input data type is int or float, the data type of `rate` will be converted to a
            1-D Tensor with paddle global default dtype.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import Poisson
            >>> paddle.set_device('cpu')
            >>> paddle.seed(100)
            >>> rv = Poisson(paddle.to_tensor(30.0))

            >>> print(rv.sample([3]))
            Tensor(shape=[3, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[35.],
             [35.],
             [30.]])

            >>> print(rv.mean)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            30.)

            >>> print(rv.entropy())
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [3.11671066])

            >>> rv1 = Poisson(paddle.to_tensor([[30.,40.],[8.,5.]]))
            >>> rv2 = Poisson(paddle.to_tensor([[1000.,40.],[7.,10.]]))
            >>> print(rv1.kl_divergence(rv2))
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[864.80285645, 0.          ],
             [0.06825157  , 1.53426421  ]])
    """

    def __init__(self, rate):
        self.dtype = paddle.get_default_dtype()
        self.rate = self._to_tensor(rate)

        if not self._check_constraint(self.rate):
            raise ValueError(
                'Every element of input parameter `rate` should be nonnegative.'
            )
        if self.rate.shape == []:
            batch_shape = (1,)
        else:
            batch_shape = self.rate.shape
        super().__init__(batch_shape)

    def _to_tensor(self, rate):
        """Convert the input parameters into tensors.

        Returns:
            Tensor: converted rate.
        """
        # convert type
        if isinstance(rate, (float, int)):
            rate = paddle.to_tensor([rate], dtype=self.dtype)
        else:
            self.dtype = rate.dtype
        return rate

    def _check_constraint(self, value):
        """Check the constraint for input parameters

        Args:
            value (Tensor)

        Returns:
            bool: pass or not.
        """
        return (value >= 0).all()

    @property
    def mean(self):
        """Mean of poisson distribution.

        Returns:
            Tensor: mean value.
        """
        return self.rate

    @property
    def variance(self):
        """Variance of poisson distribution.

        Returns:
            Tensor: variance value.
        """
        return self.rate

    def sample(self, shape=()):
        """Generate poisson samples of the specified shape. The final shape would be ``shape+batch_shape`` .

        Args:
            shape (Sequence[int], optional): Prepended shape of the generated samples.

        Returns:
            Tensor: Sampled data with shape `sample_shape` + `batch_shape`.
        """
        if not isinstance(shape, Sequence):
            raise TypeError('sample shape must be Sequence object.')

        shape = tuple(shape)
        batch_shape = tuple(self.batch_shape)
        output_shape = tuple(shape + batch_shape)
        output_rate = paddle.broadcast_to(self.rate, shape=output_shape)

        with paddle.no_grad():
            return paddle.poisson(output_rate)

    def entropy(self):
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            \mathcal{H}(X) = - \sum_{x \in \Omega} p(x) \log{p(x)}

        In the above equation:

        * :math:`\Omega`: is the support of the distribution.

        Returns:
            Tensor: Shannon entropy of poisson distribution. The data type is the same as `rate`.
        """
        values = self._enumerate_bounded_support(self.rate).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )
        log_prob = self.log_prob(values)
        proposed = -(paddle.exp(log_prob) * log_prob).sum(0)
        mask = paddle.cast(
            paddle.not_equal(
                self.rate, paddle.to_tensor(0.0, dtype=self.dtype)
            ),
            dtype=self.dtype,
        )
        return paddle.multiply(proposed, mask)

    def _enumerate_bounded_support(self, rate):
        """Generate a bounded approximation of the support. Approximately view Poisson r.v. as a
        Normal r.v. with mu = rate and sigma = sqrt(rate). Then by 30-sigma rule, generate a bounded
        approximation of the support.

        Args:
            rate (float): rate of one poisson r.v.

        Returns:
            Tensor: the bounded approximation of the support
        """
        s_max = (
            paddle.sqrt(paddle.max(rate))
            if paddle.greater_equal(
                paddle.max(rate), paddle.to_tensor(1.0, dtype=self.dtype)
            )
            else paddle.ones_like(rate, dtype=self.dtype)
        )
        upper = paddle.max(paddle.cast(rate + 30 * s_max, dtype="int32"))
        values = paddle.arange(0, upper, dtype=self.dtype)
        return values

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability. The data type is the same as `rate`.
        """
        value = paddle.cast(value, dtype=self.dtype)
        if not self._check_constraint(value):
            raise ValueError(
                'Every element of input parameter `value` should be nonnegative.'
            )
        eps = paddle.finfo(self.rate.dtype).eps
        return paddle.nan_to_num(
            (
                -self.rate
                + value * paddle.log(self.rate)
                - paddle.lgamma(value + 1)
            ),
            neginf=-eps,
        )

    def prob(self, value):
        """Probability density/mass function.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: probability. The data type is the same as `rate`.
        """
        return paddle.exp(self.log_prob(value))

    def kl_divergence(self, other):
        r"""The KL-divergence between two poisson distributions with the same `batch_shape`.

        The probability density function (pdf) is

        .. math::

            KL\_divergence\lambda_1, \lambda_2) = \sum_x p_1(x) \log{\frac{p_1(x)}{p_2(x)}}

        .. math::

            p_1(x) = \frac{e^{-\lambda_1} \cdot \lambda_1^x}{x!}

        .. math::

            p_2(x) = \frac{e^{-\lambda_2} \cdot \lambda_2^x}{x!}

        Args:
            other (Poisson): instance of ``Poisson``.

        Returns:
            Tensor, kl-divergence between two poisson distributions. The data type is the same as `rate`.

        """

        if self.batch_shape != other.batch_shape:
            raise ValueError(
                "KL divergence of two poisson distributions should share the same `batch_shape`."
            )
        rate_max = paddle.max(paddle.maximum(self.rate, other.rate))
        support_max = self._enumerate_bounded_support(rate_max)
        a_max = paddle.max(support_max)
        common_support = paddle.arange(0, a_max, dtype=self.dtype).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )

        log_prob_1 = self.log_prob(common_support)
        log_prob_2 = other.log_prob(common_support)
        return (paddle.exp(log_prob_1) * (log_prob_1 - log_prob_2)).sum(0)
