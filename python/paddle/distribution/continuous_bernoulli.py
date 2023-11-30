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


class ContinuousBernoulli(distribution.Distribution):
    r"""The Continuous Bernoulli distribution with parameter: `probability` characterizing the shape of the density function.
    The Continuous Bernoulli distribution is defined on [0, 1], and it can be viewed as a continuous version of the Bernoulli distribution.

    [1] Loaiza-Ganem, G., & Cunningham, J. P. The continuous Bernoulli: fixing a pervasive error in variational autoencoders. 2019.

    Mathematical details

    The probability density function (pdf) is

    .. math::

        p(x;\lambda) = C(\lambda)\lambda^x (1-\lambda)^{1-x}

    In the above equation:

    * :math:`x`: is continuous between 0 and 1
    * :math:`probability = \lambda`: is the probability.
    * :math:`C(\lambda)`: is the normalizing constant factor

    .. math::

        C(\lambda) =
        \left\{
        \begin{aligned}
        &2 & \text{ if $\lambda = \frac{1}{2}$} \\
        &\frac{2\tanh^{-1}(1-2\lambda)}{1 - 2\lambda} & \text{ otherwise}
        \end{aligned}
        \right.

    Args:
        probability(int|float|Tensor): The probability of Continuous Bernoulli distribution between [0, 1],
                    which characterize the shape of the pdf. The data type of `probability` will be convert to the global default dtype.
        eps(float): Specify the bandwith of the unstable calculation region near 0.5. The unstable calculation region
                    would be [0.5 - eps, 0.5 + eps], where the calculation is approximated by talyor expansion. The
                    default value is 1e-4.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distribution import ContinuousBernoulli

            # init `probability` with `paddle.Tensor`
            rv = ContinuousBernoulli(paddle.to_tensor([0.2, 0.5]))

            print(rv.sample([2]))
            # Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        [[0.09428147, 0.81438422],
            #        [0.24624705, 0.93354583]])

            print(rv.mean)
            # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        [0.38801414, 0.50000000])

            print(rv.entropy())
            # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        [-0.07641461,  0.        ])

            rv1 = ContinuousBernoulli(paddle.to_tensor([0.2, 0.8]))
            rv2 = ContinuousBernoulli(paddle.to_tensor([0.7, 0.5]))
            print(rv1.kl_divergence(rv2))
            # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        [0.20103613, 0.07641447])
    """

    def __init__(self, probability, eps=1e-4):
        self.eps = paddle.to_tensor(eps)
        self.dtype = paddle.get_default_dtype()
        self.probability = self._to_tensor(probability)

        # eps_prob is used to clip the input `probability` in the range of [eps_prob, 1-eps_prob]
        eps_prob = paddle.finfo(self.probability.dtype).eps
        self.probability = paddle.clip(
            self.probability, min=eps_prob, max=1 - eps_prob
        )

        if not self._check_constraint(self.probability):
            raise ValueError(
                'Every element of input parameter `rate` should be nonnegative.'
            )
        if self.probability.shape == []:
            batch_shape = (1,)
        else:
            batch_shape = self.probability.shape
        super().__init__(batch_shape)

    def _to_tensor(self, probability):
        """Convert the input parameters into tensors with the global default dtype

        Returns:
            Tensor: converted probability.
        """
        # convert type
        if isinstance(probability, (float, int)):
            probability = [probability]
        probability = paddle.to_tensor(probability, dtype=self.dtype)
        return probability

    def _check_constraint(self, value):
        """Check the constraint for input parameters

        Args:
            value (Tensor)

        Returns:
            bool: pass or not.
        """
        return (value >= 0).all() and (value <= 1).all()

    def _cut_support_region(self):
        """Generate stable support region indicator (prob < 0.5 - self.eps && prob >= 0.5 + self.eps )

        Returns:
            Tensor: the element of the returned indicator tensor corresponding to stable region is True, and False otherwise
        """
        return paddle.logical_or(
            paddle.less_equal(self.probability, 0.5 - self.eps),
            paddle.greater_than(self.probability, 0.5 + self.eps),
        )

    def _cut_probs(self):
        """Cut the probability parameter with stable support region

        Returns:
            Tensor: the element of the returned probability tensor corresponding to unstable region is set to be (0.5 - self.eps), and unchanged otherwise
        """
        return paddle.where(
            self._cut_support_region(),
            self.probability,
            (0.5 - self.eps) * paddle.ones_like(self.probability),
        )

    def _tanh_inverse(self, value):
        """Calculate the tanh inverse of value

        Args:
            value (Tensor)

        Returns:
            Tensor: tanh inverse of value
        """
        return 0.5 * (paddle.log1p(value) - paddle.log1p(-value))

    def _log_constant(self):
        """Calculate the logarithm of the constant factor :math:`C(lambda)` in the pdf of the Continuous Bernoulli distribution

        Returns:
            Tensor: logarithm of the constant factor
        """
        cut_probs = self._cut_probs()
        cut_probs_below_half = paddle.where(
            paddle.less_equal(cut_probs, paddle.to_tensor(0.5)),
            cut_probs,
            paddle.zeros_like(cut_probs),
        )
        cut_probs_above_half = paddle.where(
            paddle.greater_equal(cut_probs, paddle.to_tensor(0.5)),
            cut_probs,
            paddle.ones_like(cut_probs),
        )
        log_constant_propose = paddle.log(
            2.0 * paddle.abs(self._tanh_inverse(1.0 - 2.0 * cut_probs))
        ) - paddle.where(
            paddle.less_equal(cut_probs, paddle.to_tensor(0.5)),
            paddle.log1p(-2.0 * cut_probs_below_half),
            paddle.log(2.0 * cut_probs_above_half - 1.0),
        )
        x = paddle.square(self.probability - 0.5)
        taylor_expansion = (
            paddle.log(paddle.to_tensor(2.0))
            + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        )
        return paddle.where(
            self._cut_support_region(), log_constant_propose, taylor_expansion
        )

    @property
    def mean(self):
        """Mean of Continuous Bernoulli distribuion.

        Returns:
            Tensor: mean value.
        """
        cut_probs = self._cut_probs()
        tmp = paddle.divide(cut_probs, 2.0 * cut_probs - 1.0)
        propose = tmp + paddle.divide(
            paddle.to_tensor(1.0, dtype=self.dtype),
            2.0 * self._tanh_inverse(1.0 - 2.0 * cut_probs),
        )
        x = self.probability - 0.5
        taylor_expansion = (
            0.5 + (1.0 / 3.0 + 16.0 / 45.0 * paddle.square(x)) * x
        )
        return paddle.where(
            self._cut_support_region(), propose, taylor_expansion
        )

    @property
    def variance(self):
        """Variance of Continuous Bernoulli distribution.

        Returns:
            Tensor: variance value.
        """
        cut_probs = self._cut_probs()
        tmp = paddle.divide(
            paddle.square(cut_probs) - cut_probs,
            paddle.square(1.0 - 2.0 * cut_probs),
        )
        propose = tmp + paddle.divide(
            paddle.to_tensor(1.0, dtype=self.dtype),
            paddle.square(2.0 * self._tanh_inverse(1.0 - 2.0 * cut_probs)),
        )
        x = paddle.square(self.probability - 0.5)
        taylor_expansion = 1.0 / 12.0 - (1.0 / 15.0 - 128.0 / 945.0 * x) * x
        return paddle.where(
            self._cut_support_region(), propose, taylor_expansion
        )

    def sample(self, shape=()):
        """Generate Continuous Bernoulli samples of the specified shape. The final shape would be ``sample_shape + batch_shape``.

        Args:
            shape (Sequence[int], optional): Prepended shape of the generated samples.

        Returns:
            Tensor, Sampled data with shape `sample_shape` + `batch_shape`. The data type is the global default dtype.
        """
        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=()):
        """Generate Continuous Bernoulli samples of the specified shape. The final shape would be ``sample_shape + batch_shape``.

        Args:
            shape (Sequence[int], optional): Prepended shape of the generated samples.

        Returns:
            Tensor, Sampled data with shape `sample_shape` + `batch_shape`. The data type is the global default dtype.
        """
        if not isinstance(shape, Sequence):
            raise TypeError('sample shape must be Sequence object.')
        shape = tuple(shape)
        batch_shape = tuple(self.batch_shape)
        output_shape = tuple(shape + batch_shape)
        u = paddle.uniform(shape=output_shape, dtype=self.dtype, min=0, max=1)
        return self.icdf(u)

    def log_prob(self, value):
        """Log probability density function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability. The data type is same with :attr:`value` .
        """
        value = paddle.cast(value, dtype=self.dtype)
        if not self._check_constraint(value):
            raise ValueError(
                'Every element of input parameter `value` should be >= 0.0 and <= 1.0.'
            )
        eps = paddle.finfo(self.probability.dtype).eps
        cross_entropy = paddle.nan_to_num(
            value * paddle.log(self.probability)
            + (1.0 - value) * paddle.log(1 - self.probability),
            neginf=-eps,
        )
        return self._log_constant() + cross_entropy

    def prob(self, value):
        """Probability density function.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: probability. The data type is same with :attr:`value` .
        """
        return paddle.exp(self.log_prob(value))

    def entropy(self):
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            \mathcal{H}(X) = -\log C + \left[ \log (1 - \lambda) -\log \lambda \right] \mathbb{E}(X)  - \log(1 - \lambda)

        In the above equation:

        * :math:\Omega: is the support of the distribution.

        Returns:
            Tensor, Shannon entropy of Continuous Bernoulli distribution. The data type is the global default dtype.
        """
        log_p = paddle.log(self.probability)
        log_1_minus_p = paddle.log1p(-self.probability)

        return (
            -self._log_constant()
            + self.mean * (log_1_minus_p - log_p)
            - log_1_minus_p
        )

    def cdf(self, value):
        r"""Cumulative distribution function

        .. math::

            {   P(X \le t; \lambda) =
                F(t;\lambda) =
                \left\{
                \begin{aligned}
                &t & \text{ if $\lambda = \frac{1}{2}$} \\
                &\frac{\lambda^t (1 - \lambda)^{1 - t} + \lambda - 1}{2\lambda - 1} & \text{ otherwise}
                \end{aligned}
                \right. }

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: quantile of :attr:`value`. The data type is same with :attr:`value` .
        """
        value = paddle.cast(value, dtype=self.dtype)
        if not self._check_constraint(value):
            raise ValueError(
                'Every element of input parameter `value` should be >= 0.0 and <= 1.0.'
            )
        cut_probs = self._cut_probs()
        cdfs = (
            paddle.pow(cut_probs, value)
            * paddle.pow(1.0 - cut_probs, 1.0 - value)
            + cut_probs
            - 1.0
        ) / (2.0 * cut_probs - 1.0)
        unbounded_cdfs = paddle.where(self._cut_support_region(), cdfs, value)
        return paddle.where(
            paddle.less_equal(value, paddle.to_tensor(0.0)),
            paddle.zeros_like(value),
            paddle.where(
                paddle.greater_equal(value, paddle.to_tensor(1.0)),
                paddle.ones_like(value),
                unbounded_cdfs,
            ),
        )

    def icdf(self, value):
        r"""Inverse cumulative distribution function

        .. math::

            {   F^{-1}(x;\lambda) =
                \left\{
                \begin{aligned}
                &x & \text{ if $\lambda = \frac{1}{2}$} \\
                &\frac{\log(1+(\frac{2\lambda - 1}{1 - \lambda})x)}{\log(\frac{\lambda}{1-\lambda})} & \text{ otherwise}
                \end{aligned}
                \right. }

        Args:
            value (Tensor): The input tensor, meaning the quantile.

        Returns:
            Tensor: the value of the r.v. corresponding to the quantile. The data type is same with :attr:`value` .
        """
        value = paddle.cast(value, dtype=self.dtype)
        if not self._check_constraint(value):
            raise ValueError(
                'Every element of input parameter `value` should be >= 0.0 and <= 1.0.'
            )
        cut_probs = self._cut_probs()
        return paddle.where(
            self._cut_support_region(),
            (
                paddle.log1p(-cut_probs + value * (2.0 * cut_probs - 1.0))
                - paddle.log1p(-cut_probs)
            )
            / (paddle.log(cut_probs) - paddle.log1p(-cut_probs)),
            value,
        )

    def kl_divergence(self, other):
        r"""The KL-divergence between two Continuous Bernoulli distributions with the same `batch_shape`.

        The probability density function (pdf) is

        .. math::

            KL\_divergence(\lambda_1, \lambda_2) = - H - \{\log C_2 + [\log \lambda_2 -  \log (1-\lambda_2)]  \mathbb{E}_1(X) +  \log (1-\lambda_2)  \}

        Args:
            other (ContinuousBernoulli): instance of Continuous Bernoulli.

        Returns:
            Tensor, kl-divergence between two Continuous Bernoulli distributions. The data type is the global default dtype.

        """

        if self.batch_shape != other.batch_shape:
            raise ValueError(
                "KL divergence of two Continuous Bernoulli distributions should share the same `batch_shape`."
            )
        part1 = -self.entropy()
        log_q = paddle.log(other.probability)
        log_1_minus_q = paddle.log1p(-other.probability)
        part2 = -(
            other._log_constant()
            + self.mean * (log_q - log_1_minus_q)
            + log_1_minus_q
        )
        return part1 + part2
