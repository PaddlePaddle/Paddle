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
from collections.abc import Iterable

import numpy as np

import paddle
from paddle.distribution import distribution
from paddle.framework import in_dynamic_mode


class Binomial(distribution.Distribution):
    r"""The Binomial distribution with size `total_count` and `probability` parameters.

    Mathematical details

    The probability mass function (pmf) is

    .. math::

        pmf(x; n, p) = \frac{n!}{x!(n-x)!}p^{x}(1-p)^{n-x}

    In the above equation:

    * :math:`total_count = n`: is the size.
    * :math:`probability = p`: is the probability.

    Args:
        total_count(int|Tensor): The size of Binomial distribution, meaning the number of independent bernoulli
                                trials with probability parameter p. The random variable counts the number of success
                                among n independent bernoulli trials. The data type of `total_count` will be convert to float32.
        probability(float|Tensor): The probability of Binomial distribution, meaning the probability of success
                                for each individual bernoulli trial. The data type of `probability` will be convert to float32.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import Binomial
            >>> rv = Binomial(100, paddle.to_tensor([0.3, 0.6, 0.9]))

            >>> # doctest: +SKIP
            >>> print(rv.sample([2]))
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[33., 56., 93.],
            [32., 53., 91.]])

            >>> print(rv.mean)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [30.00000191, 60.00000381, 90.        ])

            >>> print(rv.entropy())
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2.94057941, 3.00785327, 2.51125669])
    """

    def __init__(self, total_count, probability):
        self.dtype = 'float32'
        self.total_count, self.probability = self._to_tensor(
            total_count, probability
        )

        if not self._check_constraint(self.total_count, self.probability):
            raise ValueError(
                'Every element of input parameter `total_count` should be grater than or equal to one, and `probability` should be grater than or equal to zero and less than or equal to one.'
            )
        if self.total_count.shape == []:
            batch_shape = (1,)
        else:
            batch_shape = self.total_count.shape
        super().__init__(batch_shape)

    def _to_tensor(self, total_count, probability):
        """Convert the input parameters into tensors and broadcast them

        Returns:
            (Tensor, Tensor): converted total_count and probability.
        """
        # convert type
        if isinstance(total_count, int):
            total_count = paddle.to_tensor([total_count], dtype=self.dtype)
        if isinstance(probability, float):
            probability = paddle.to_tensor([probability], dtype=self.dtype)
        if isinstance(total_count, np.ndarray):
            total_count = paddle.to_tensor(total_count)
        if isinstance(probability, np.ndarray):
            probability = paddle.to_tensor(probability)
        total_count = paddle.cast(total_count, dtype=self.dtype)
        probability = paddle.cast(probability, dtype=self.dtype)

        # broadcast tensor
        return paddle.broadcast_tensors([total_count, probability])

    def _check_constraint(self, total_count, probability):
        """Check the constraints for input parameters

        Args:
            total_count (Tensor)
            probability (Tensor)

        Returns:
            bool: pass or not.
        """
        total_count_check = (total_count >= 1).all()
        probability_check = (probability >= 0).all() * (probability <= 1).all()
        return total_count_check and probability_check

    @property
    def mean(self):
        """Mean of binomial distribuion.

        Returns:
            Tensor: mean value.
        """
        return self.total_count * self.probability

    @property
    def variance(self):
        """Variance of binomial distribution.

        Returns:
            Tensor: variance value.
        """
        return self.total_count * self.probability * (1 - self.probability)

    def sample(self, shape=()):
        """Generate binomial samples of the specified shape.

        Args:
            shape (Sequence[int], optional): Prepended shape of the generated samples.

        Returns:
            Tensor, A tensor with prepended dimensions shape. The data type is float32.
        """
        if not isinstance(shape, Iterable):
            raise TypeError('sample shape must be Iterable object.')

        with paddle.set_grad_enabled(False):
            shape = tuple(shape)
            batch_shape = tuple(self.batch_shape)
            output_shape = tuple(shape + batch_shape)
            output_size = paddle.broadcast_to(
                self.total_count, shape=output_shape
            )
            output_prob = paddle.broadcast_to(
                self.probability, shape=output_shape
            )
            if in_dynamic_mode():
                broadcast_func = np.frompyfunc(binomial_sample, nin=2, nout=1)
                return paddle.to_tensor(
                    broadcast_func(
                        output_size.numpy(), output_prob.numpy()
                    ).astype('float32')
                )
            else:
                output = (
                    paddle.static.default_main_program()
                    .current_block()
                    .create_var(dtype=self.dtype, shape=output_shape)
                )
                paddle.static.py_func(
                    func=binomial_sample_vectorized,
                    x=[output_size, output_prob],
                    out=output,
                )
                return output

    def entropy(self):
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            \mathcal{H}(X) = - \sum_{x \in \Omega} p(x) \log{p(x)}

        In the above equation:

        * :math:\Omega: is the support of the distribution.

        Returns:
            Tensor: Shannon entropy of binomial distribution. The data type is float32.
        """
        """Evaluated the entropy of one binomial r.v..

        Args:
            n (float): size of the binomial r.v.
            p (float): probability of the binomial r.v.

        Returns:
            numpy.ndarray: the entropy for the binomial r.v.
        """
        values = self._enumerate_support()
        log_prob = self.log_prob(values)
        return -(paddle.exp(log_prob) * log_prob).sum(0)

    def _enumerate_support(self):
        """Return the support of binomial distribution [0, 1, ... ,n]

        Returns:
            Tensor: the support of binomial distribution
        """
        values = paddle.arange(
            1 + paddle.max(self.total_count), dtype=self.dtype
        )
        values = values.reshape((-1,) + (1,) * len(self.batch_shape))
        return values

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability. The data type is same with :attr:`value` .
        """
        value = paddle.cast(value, dtype=self.dtype)

        # combination
        log_comb = (
            paddle.lgamma(self.total_count + 1.0)
            - paddle.lgamma(self.total_count - value + 1.0)
            - paddle.lgamma(value + 1.0)
        )
        eps = paddle.finfo(self.probability.dtype).eps
        probs = paddle.clip(self.probability, min=eps, max=1 - eps)
        # log_p
        return paddle.nan_to_num(
            (
                log_comb
                + value * paddle.log(probs)
                + (self.total_count - value) * paddle.log(1 - probs)
            ),
            neginf=-eps,
        )

    def prob(self, value):
        """Probability density/mass function.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: probability. The data type is same with :attr:`value` .
        """
        return paddle.exp(self.log_prob(value))

    def kl_divergence(self, other):
        r"""The KL-divergence between two binomial distributions.

        The probability density function (pdf) is

        .. math::

            KL\_divergence(n_1, p_1, n_2, p_2) = \sum_x p_1(x) \log{\frac{p_1(x)}{p_2(x)}}

        .. math::

            p_1(x) = \frac{n_1!}{x!(n_1-x)!}p_1^{x}(1-p_1)^{n_1-x}

        .. math::

            p_2(x) = \frac{n_2!}{x!(n_2-x)!}p_2^{x}(1-p_2)^{n_2-x}

        Args:
            other (Binomial): instance of Binomial.

        Returns:
            Tensor: kl-divergence between two binomial distributions. The data type is float32.

        """
        if not (paddle.equal(self.total_count, other.total_count)).all():
            raise ValueError(
                "KL divergence of two binomial distributions should share the same `total_count` and `batch_shape`."
            )
        support = self._enumerate_support()
        log_prob_1 = self.log_prob(support)
        log_prob_2 = other.log_prob(support)
        return (
            paddle.multiply(
                paddle.exp(log_prob_1),
                (paddle.subtract(log_prob_1, log_prob_2)),
            )
        ).sum(0)


def small_binomial_sample(f, p_q, g, n, p):
    """Binomial r.v. sampling algorithm when size*probability < 30. By inverse cdf method.

    Returns:
        int, one binomial sample.
    """
    while True:
        y = 0
        u = np.random.uniform(0, 1)
        while True:
            if u < f:
                return n - y if p > 0.5 else y
            if y > 110:
                break
            u -= f
            y += 1
            f *= g / y - p_q


def binomial_sample(n, p):
    """BTPE algorithm, a Binomial r.v. sampling algorithm.

    Args:
        n (float): size
        p (float): probability

    Returns:
        int, one binomial sample.

    ### References

    [1]: Kachitvichyanukul, V. and Schmeiser, B. W. (1988) Binomial random variate generation.
            Communications of the ACM, 31, 216–222.
    """

    if not in_dynamic_mode():
        n = np.array(n)
        p = np.array(p)
    if p == 1:
        return n
    if n == 0 or p == 0:
        return 0
    r = min(p, 1 - p)
    q = 1 - r
    p_q = r / q
    g = p_q * (n + 1)

    # small n*r
    if n * r < 30:
        f = math.pow(q, n)
        return small_binomial_sample(f, p_q, g, n, p)
    else:
        # step 0
        f_M = n * r + r
        M = int(f_M)
        p_1 = int(2.195 * math.sqrt(n * r * q) - 4.6 * q) + 0.5
        x_M = M + 0.5
        x_L = x_M - p_1
        x_R = x_M + p_1
        c = 0.134 + 20.5 / (15.3 + M)
        a = (f_M - x_L) / (f_M - x_L * r)
        l_L = a * (1 + a / 2)
        try:
            a = (x_R - f_M) / (x_R * q)
        except:
            a = math.inf
        l_R = a * (1 + a / 2)
        p_2 = p_1 * (1 + 2 * c)
        p_3 = p_2 + c / l_L
        p_4 = p_3 + c / l_R
    while True:
        # step 1
        u = np.random.uniform(0, 1) * p_4
        v = np.random.uniform(0, 1)
        if u <= p_1:
            y = int(x_M - p_1 * v + u)
            return n - y if p > 0.5 else y
        # step 2
        if u <= p_2:
            x = x_L + (u - p_1) / c
            v = v * c + 1 - abs(x_M - x) / p_1
            if v > 1 or v <= 0:
                continue
            y = int(x)
        else:
            # step 3
            if u > p_3:
                # step 4
                y = int(x_R - math.log(v) / l_R)
                if y > n:
                    continue
                v = v * (u - p_3) * l_R
            else:
                y = int(x_L + math.log(v) / l_L)
                if y < 0:
                    continue
                v = v * (u - p_2) * l_L

        # step 5
        k = abs(y - M)
        if k <= 20 or k >= (n * r * q) / 2 - 1:
            s = r / q
            a = s * (n + 1)
            F = 1.0
            if M < y:
                for i in range(M + 1, y + 1):
                    F = F * (a / i - s)
            elif M > y:
                for i in range(y + 1, M + 1):
                    try:
                        F = F / (a / i - s)
                    except:
                        F = math.inf
            if v <= F:
                return n - y if p > 0.5 else y
            else:
                continue
        else:
            rho = (k / (n * r * q)) * (
                (k * (k / 3 + 0.625) + 1 / 6) / (n * r * q) + 0.5
            )
            t = -(k**2) / (2 * (n * r * q))
            A = math.log(v)
            if A < t - rho:
                return n - y if p > 0.5 else y
            if A <= t + rho:
                # 5.3
                x_1 = y + 1
                f_1 = M + 1
                z = n + 1 - M
                w = n - y + 1
                x_2 = x_1**2
                f_2 = f_1**2
                z_2 = z**2
                w_2 = w**2
                tmp = (
                    x_M * math.log(f_1 / x_1)
                    + (n - M + 0.5) * math.log(z / w)
                    + (y - M) * math.log(w * r / (x_1 * q))
                )
                tmp += (
                    (
                        13860.0
                        - (462.0 - (132.0 - (99.0 - 140.0 / f_2) / f_2) / f_2)
                        / f_2
                    )
                    / f_1
                    / 166320.0
                )
                tmp += (
                    (
                        13860.0
                        - (462.0 - (132.0 - (99.0 - 140.0 / z_2) / z_2) / z_2)
                        / z_2
                    )
                    / z
                    / 166320.0
                )
                tmp += (
                    (
                        13860.0
                        - (462.0 - (132.0 - (99.0 - 140.0 / x_2) / x_2) / x_2)
                        / x_2
                    )
                    / x_1
                    / 166320.0
                )
                tmp += (
                    (
                        13860.0
                        - (462.0 - (132.0 - (99.0 - 140.0 / w_2) / w_2) / w_2)
                        / w_2
                    )
                    / w
                    / 166320.0
                )
                if A <= tmp:
                    return n - y if p > 0.5 else y


def binomial_sample_vectorized(nn, pp):
    """Vectorized binomial sampling function

    Args:
        nn (Tensor): size
        pp (Tensor): probability

    Returns:
        np.ndarray, binomial samples.
    """
    broadcast_func = np.frompyfunc(binomial_sample, nin=2, nout=1)
    return np.array(broadcast_func(nn, pp), dtype='float32')
