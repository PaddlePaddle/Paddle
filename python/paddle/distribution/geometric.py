# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distribution import distribution
from paddle.distribution import uniform
from paddle.fluid import framework


class Geometric(distribution.Distribution):
    r"""
    Geometric distribution parameterized by probs.

    In probability theory and statistics, the geometric distribution is one of
    discrete probability distributions, parameterized by one positive shape parameter, denoted by probs.
    In n Bernoulli trials, it takes k trials to get the probability of success for the first time.
    In detail, it is: the probability that the first k-1 times failed and the kth time succeeded.
    The geometric distribution is a special case of the Pascal distribution when r=1.

    The probability mass function (pmf) is

    .. math::
            Pr(Y=k)=(1-p)^kp

    where k is number of trials performed and p is probability of success for each trial and k=0,1,2,3,4..., p belong to (0,1].

    Args:
        probs (Real|Tensor): Probability parameter.
            The value of probs must be positive. When the parameter is a tensor, probs is probability of success for each trial.

    Examples:

        .. code-block:: python

            import paddle

            # probability input
            probs = paddle.full([1], fill_value=0.2, dtype=paddle.float32)
            dist = paddle.distribution.Geometric(probs)

            dist.mean
            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,[5.])
            dist.variance
            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,[20.])
            dist.stddev
            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,[4.47213602])
            dist.sample((2, 2))
            # Tensor(shape=[2, 2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,[[[2.09490776],[0.00561932]],[[0.17222938],[5.50346899]]])
            dist.rsample((2, 2))
            # Tensor(shape=[2, 2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,[[[6.05133677],[9.59971237]],[[1.72454977],[0.30767119]]])
            dist.pmf(2)
            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,[0.16000001])
            dist.log_pmf(2)
            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,[-1.83258140])
            dist.cdf(4)
            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [0.59039998])
            dist.entropy()
            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,[2.50201201])
            other_probs = paddle.full([1], fill_value=0.3, dtype=paddle.float32)
            other_dist = paddle.distribution.Geometric(other_probs)
            dist.kl_divergence(other_dist)
            # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [0.08109304])

    """

    def __init__(self, probs=None):
        if isinstance(probs, (numbers.Real, paddle.Tensor, framework.Variable)):
            if isinstance(probs, numbers.Real):
                probs = paddle.full(
                    shape=(1,), fill_value=probs, dtype=paddle.float32
                )

            all_ones = paddle.full(
                shape=probs.shape, fill_value=1, dtype=probs.dtype
            )
            all_zeros = paddle.full(
                shape=probs.shape, fill_value=0, dtype=probs.dtype
            )
            all_false = paddle.full(
                shape=probs.shape, fill_value=False, dtype=bool
            )

            lessthen_0 = probs <= all_zeros
            morethen_1 = probs > all_ones

        else:
            raise TypeError(
                f"Expected type of probs is Number.Real|Tensor|framework.Variable, but got {type(probs)}"
            )

        if paddle.equal_all(lessthen_0, all_false) and paddle.equal_all(
                morethen_1, all_false
        ):
            batch_shape = tuple(probs.shape)
        else:
            raise ValueError(
                "Expected parameter probs of distribution Geometric to satisfy the"
                "constraint Interval(lower_bound=0.0, upper_bound=1.0)"
            )

        self.probs = probs
        super(Geometric, self).__init__(batch_shape)

    @property
    def mean(self):
        """Mean of geometric distribution."""
        return 1.0 / self.probs

    @property
    def variance(self):
        """Variance of geometric distribution"""
        return paddle.to_tensor(
            (1.0 / self.probs - 1.0) / self.probs,
            dtype=self.probs.dtype,
        )

    @property
    def stddev(self):
        """Standard deviation of Geometric distribution"""
        return paddle.sqrt(self.variance)

    def pmf(self, k):
        """Probability mass funciotn evaluated at k

        Args:
            k (int): Value to be evaluated.

        Returns:
            Tensor: Probability.
        """
        if isinstance(k, (numbers.Integral, framework.Variable)):
            return paddle.pow((1.0 - self.probs), k - 1.0) * self.probs
        else:
            raise TypeError(f"Expected type of k is number.Real|framework.Variable, but got {type(k)}")

    def log_pmf(self, k):
        """Log probability mass function evaluated at k

        Args:
            k (int): Value to be evaluated

        Returns:
            Tensor: Log probability.
        """
        if isinstance(k, (numbers.Integral, framework.Variable)):
            return paddle.log(self.pmf(k))
        else:
            raise TypeError(f"Expected type of k is number.Real|framework.Variable, but got {type(k)}")

    def sample(self, shape=()):
        """Sample from Geometric distribution with sample shape.

        Args:
            shape (Sequence[int], optional): Sample shape.

        Returns:
            Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`.
        """
        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=()):
        """Generate samples of the specified shape.

        Args:
            shape(tuple): The shape of generated samples.

        Returns:
            Tensor: A sample tensor that fits the Geometric distribution.
        """
        shape = distribution.Distribution._extend_shape(self, sample_shape=shape)
        tiny = np.finfo(dtype='float32').tiny

        sample_uniform = uniform.Uniform(low=float(tiny), high=float(1))

        new_t = sample_uniform.sample(list(shape))
        return paddle.log(new_t) / paddle.log1p(-(self.probs))

    def entropy(self):
        """Entropy of dirichlet distribution

        Returns:
            Tensor: Entropy.
        """
        x = (1.0 - self.probs) * paddle.log(
            1.0 - self.probs
        )
        y = self.probs * paddle.log(self.probs)

        return -(x + y) / self.probs

    def cdf(self, k):
        """Cdf of geometric distribution

        Args:
            k: The number of trials performed.

        Returns:
            Tensor: Entropy.
        """
        if isinstance(k, (numbers.Integral, framework.Variable)):
            return 1.0 - paddle.pow((1.0 - self.probs), k)
        else:
            raise TypeError(f"Expected type of k is number.Real|framework.Variable, but got {type(k)}")

    def kl_divergence(self, other):
        """Calculate the KL divergence KL(self || other) with two Geometric instances.

        Args:
            other (Geometric): An instance of Geometric.

        Returns:
            Tensor: The kl-divergence between two geometric distributions.
        """
        if isinstance(other, Geometric):
            p, q = self.probs, other.probs
            return p * paddle.log(p / q) + (1.0 - p) * paddle.log((1.0 - p) / (1.0 - q))
        else:
            raise TypeError(f"Exected type of other is geometric.Geometric, but got {type(other)}")
