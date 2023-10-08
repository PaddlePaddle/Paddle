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


class Poisson(distribution.Distribution):
    r"""The Poisson distribution with occurrence rate parameter: `rate`.

    Mathematical details

    The probability density function (pdf) is

    .. math::

        pdf(x; \lambda) = \frac{e^{-\lambda} \cdot \lambda^x}{x!}$

    In the above equation:

    * :math:`rate = \lambda`: is the mean occurrence rate.

    Args:
        rate(int|float|Tensor): The mean occurrence rate of Poisson distribution, meaning the expected occurrence
                                times of an event in a fixed time interval. The data type of `rate` will be convert to float32.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import Binomial
            >>> rv = Poisson(paddle.to_tensor(30))
            >>> print(rv.sample([3, 4]))
            Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                [[28., 39., 22., 35.],
                [36., 34., 47., 22.],
                [28., 24., 33., 31.]])
            >>> print(rv.mean)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                30.)
            >>> print(rv.entropy())
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [3.11661315])
            >>> rv1 = Poisson(paddle.to_tensor([[30,40],[8,5]]))
            >>> rv2 = Poisson(paddle.to_tensor([[1000,40],[7,10]]))
            >>> print(rv1.kl_divergence(rv2))
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
                [[864.80285645, 0.          ],
                [0.06825157  , 1.53426421  ]])
    """

    def __init__(self, rate):
        self.dtype = 'float32'
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
        """Convert the input parameters into tensors with dtype = float32

        Returns:
            Tensor: converted rate.
        """
        # convert type
        if isinstance(rate, (float, int)):
            rate = paddle.to_tensor([rate], dtype=self.dtype)
        if isinstance(rate, np.ndarray):
            rate = paddle.to_tensor(rate)
        rate = paddle.cast(rate, dtype=self.dtype)
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
        """Mean of poisson distribuion.

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
        """Generate poisson samples of the specified shape.

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
            output_rate = paddle.broadcast_to(self.rate, shape=output_shape)
            if in_dynamic_mode():
                broadcast_func = np.frompyfunc(poisson_sample, nin=1, nout=1)
                return paddle.to_tensor(
                    broadcast_func(output_rate.numpy()).astype('float32')
                )
            else:
                output = (
                    paddle.static.default_main_program()
                    .current_block()
                    .create_var(dtype=self.dtype, shape=output_shape)
                )
                paddle.static.py_func(
                    func=poisson_sample_vectorized,
                    x=output_rate,
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
            Tensor, Shannon entropy of poisson distribution. The data type is float32.
        """
        values = self._enumerate_bounded_support(self.rate).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )
        log_prob = self.log_prob(values)
        return -(paddle.exp(log_prob) * log_prob).sum(0)

    def _enumerate_bounded_support(self, rate):
        """Generate a bounded approximation of the support. Approximately view Poisson r.v. as a Normal r.v. with mu = rate and sigma = sqrt(rate).
        Then by 30-sigma rule, generate a bounded approximation of the support.

        Args:
            rate (float): rate of one poisson r.v.

        Returns:
            numpy.ndarray: the bounded approximation of the support
        """
        s_max = paddle.sqrt(paddle.max(rate))
        upper = paddle.max(paddle.cast(rate + 30 * s_max, dtype="int32"))
        values = paddle.arange(0, upper, dtype="float32")
        return values

    def log_prob(self, value):
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability. The data type is same with :attr:`value` .
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
            neginf=eps,
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
        r"""The KL-divergence between two poisson distributions.

        The probability density function (pdf) is

        .. math::

            KL\_divergence\lambda_1, \lambda_2) = \sum_x p_1(x) \log{\frac{p_1(x)}{p_2(x)}}

        .. math::

            p_1(x) = \frac{e^{-\lambda_1} \cdot \lambda_1^x}{x!}

        .. math::

            p_2(x) = \frac{e^{-\lambda_2} \cdot \lambda_2^x}{x!}

        Args:
            other (Poisson): instance of Poisson.

        Returns:
            Tensor, kl-divergence between two poisson distributions. The data type is float32.

        """

        if self.batch_shape != other.batch_shape:
            raise ValueError(
                "KL divergence of two poisson distributions should share the same `batch_shape`."
            )
        rate_max = paddle.max(paddle.maximum(self.rate, other.rate))
        rate_min = paddle.min(paddle.minimum(self.rate, other.rate))
        support_max = self._enumerate_bounded_support(rate_max)
        support_min = self._enumerate_bounded_support(rate_min)
        a_min = paddle.min(support_min)
        a_max = paddle.max(support_max)
        common_support = paddle.arange(a_min, a_max, dtype=self.dtype).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )

        log_prob_1 = self.log_prob(common_support)
        log_prob_2 = other.log_prob(common_support)
        return (paddle.exp(log_prob_1) * (log_prob_1 - log_prob_2)).sum(0)


def poisson_sample(rate):
    """PD algorithm, a Poisson r.v. sampling algorithm.

    Args:
        rate (float): rate

    Returns:
        int, one poisson sample

    ### References

    [1]: Ahrens, J. H. and Dieter, U. (1982). Computer generation of Poisson deviates
            from modified normal distributions. ACM Transactions on Mathematical Software, 8, 163–179.
    """
    if rate == 0:
        return 0
    factorial_table = np.array(
        [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0]
    )  # factorial of 0~9
    error_table = np.array(
        [
            -0.5,
            0.3333333,
            -0.2500068,
            0.2000118,
            -0.1661269,
            0.1421878,
            -0.1384794,
            0.1250060,
        ]
    )  # truncation error table with |\epsilon| < 2*10^{-8} for taylor expension of equation (6) in the paper
    p_table = np.zeros([36])

    big_rate = rate > 10
    rate_prev = 0
    rate_prev2 = 0
    rate_changed = False

    def _step_F():
        # step F
        if K < 10:
            px = -rate
            py = rate**K / factorial_table[int(K)]
        else:
            d = 1 / (12 * K)
            d = d - 4.8 * d**3
            V = (rate - K) / K
            if abs(V) <= 0.25:
                sum_av = 1
                for i in range(len(error_table)):
                    sum_av += error_table[i] * V**i
                px = K * V**2 * sum_av - d
            else:
                px = K * math.log(1 + V) - (rate - K) - d
            py = 1 / math.sqrt(2 * math.pi) / math.sqrt(K)
        X = (K - rate + 0.5) / s
        X *= X
        fx = -0.5 * X
        fy = w * (((c_3 * X + c_2) * X + c_1) * X + c_0)
        return px, py, fx, fy

    if not (big_rate and rate == rate_prev):
        if big_rate:  # rate >= 10, and rate changed
            # case A
            rate_changed = True
            rate_prev = rate
            s = math.sqrt(rate)
            d = 6 * rate**2
            L = int(rate - 1.1484)
        else:  # rate < 10
            # case B
            if rate != rate_prev:
                rate_prev = rate
                M = max(1, int(rate))
                L2 = 0
                p = q = p_0 = math.exp(-rate)
            while True:
                # step U
                u = np.random.uniform()
                if u <= p_0:
                    return 0
                # step T
                if L2 != 0:
                    for k in range(1 if u <= 0.458 else min(L2, M), L2 + 1):
                        if u <= p_table[k]:
                            return k
                    if L2 == 35:
                        continue
                # step C
                for k in range(L2 + 1, 36):
                    p = p * rate / k
                    q += p
                    p_table[k] = q
                    if u <= q:
                        L2 = k
                        return k
                L2 = 35
    # step N
    G = rate + s * np.random.standard_normal()
    if G >= 0:
        K = int(G)
        if K >= L:  # step I
            return K
        u = np.random.uniform()  # step S
        if d * u >= (rate - K) ** 3:
            return K

    # step P
    if rate_changed or rate != rate_prev2:
        rate_prev2 = rate
        w = 1 / math.sqrt(2 * math.pi) / s
        b_1 = 1 / 24 / rate
        b_2 = 3 / 10 * b_1**2
        c_3 = 1 / 7 * b_1 * b_2
        c_2 = b_2 - 15 * c_3
        c_1 = b_1 - 6 * b_2 + 45 * c_3
        c_0 = 1 - b_1 + 3 * b_2 - 15 * c_3
        c = 0.1069 / rate
    if G >= 0:  # G >= 0: F -> Q? -> E -> E... -> E -> F -> H
        px, py, fx, fy = _step_F()
        # step Q
        if fy * (1 - u) <= py * math.exp(px - fx):
            return K
        else:
            while True:
                # step E
                E = np.random.exponential()
                u = 2 * np.random.uniform() - 1
                T = 1.8 + E * np.sign(u)
                if T > 0.6744:
                    K = int(rate + s * T)
                    px, py, fx, fy = _step_F()
                    # step H
                    if c * abs(u) <= py * math.exp(px + E) - fy * math.exp(
                        fx + E
                    ):
                        break
    else:  # G < 0: E -> E... -> E -> F -> H
        while True:
            # step E
            E = np.random.exponential()
            u = 2 * np.random.uniform() - 1
            T = 1.8 + E * np.sign(u)
            if T > 0.6744:
                K = int(rate + s * T)
                px, py, fx, fy = _step_F()
                # step H
                if c * abs(u) <= py * math.exp(px + E) - fy * math.exp(fx + E):
                    break
    return K


def poisson_sample_vectorized(rr):
    """Vectorized binomial sampling function

    Args:
        nn (Tensor): size
        pp (Tensor): probability

    Returns:
        np.ndarray, binomial samples.
    """
    broadcast_func = np.frompyfunc(poisson_sample, nin=1, nout=1)
    return np.array(broadcast_func(rr), dtype='float32')
