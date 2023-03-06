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

from paddle.distribution import Distribution
from paddle.distribution.uniform import Uniform as Un
from paddle.fluid import framework


class Geometric(Distribution):
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
        probs (float|Tensor): Probability parameter.
            The value of probs must be positive. When the parameter is a tensor, probs is probability of success for each trial.

    Examples:

        .. code-block:: python

          import paddle
          from paddle.distribution.geometric import Geometric

          # probability input
          probs = paddle.full(shape=(),file_value=0.2, dtype=paddle.float32)
          dist = Geometric(probs)
          dist.sample((2,2))
          #Tensor(shape=[2, 2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [[[0.82210195],
                     [3.06319213]],

                    [[2.74386835],
                     [2.70396948]]])
          dist.rsample((2,2))
          Tensor(shape=[2, 2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [[[9.46812630 ],
                     [4.16647911 ]],

                    [[0.80256897 ],
                     [20.91611290]]])
          dist.pmf(2)
          # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,[0.16000001])
          dist.log_pmf(2)
          # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,[-1.83258140])
          dist.cdf(4)
          # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [0.59039998])
          dist.entropy()
          # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,[2.50201201])
          other_probs = paddle.full(shape=(),fill_value=0.3, dtype=paddle.float32)
          other_dist = Geometric(other_probs)
          # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [0.08109304])

    """

    def __init__(self, probs=None):
        if (probs is None):
            raise ValueError("`Probs` must be specified.")
        if isinstance(probs, numbers.Real) or isinstance(probs, list):
             probs = paddle.full(shape=(),fill_value=probs, dtype=paddle.float32)

        if isinstance(probs, (paddle.Tensor, framework.Variable)):
            if (probs.dtype != paddle.float32):
                raise ValueError(f"Probs'dtype must be float32, but get {probs.dtype}")

            all_ones = paddle.full(shape=probs.shape, fill_value=1, dtype=probs.dtype)
            all_zeros = paddle.full(shape=probs.shape, fill_value=0, dtype=probs.dtype)
            all_false = paddle.full(shape=probs.shape, fill_value=False, dtype=bool)

            lessthen_0 = probs > all_ones
            morethen_1 = probs <= all_zeros

            if (paddle.equal_all(lessthen_0, all_false) and paddle.equal_all(morethen_1, all_false)):
                batch_shape = tuple(probs.shape)
            else:
                raise ValueError(
                    f"Expected parameter probs of distribution Geometric to satisfy the "
                    f"constraint Interval(lower_bound=0.0, upper_bound=1.0)"
                )
        else:
            raise TypeError(
                f"Expected type of probs is Number|List|Tensor, but got {type(probs)}"
            )

        self.probs = probs
        self.one_tensor = paddle.full(self.probs.shape,
                                      fill_value=1,
                                      dtype=self.probs.dtype)
        super(Geometric, self).__init__(batch_shape)

    @property
    def mean(self):
        """Mean of geometric distribution."""
        return self.one_tensor / self.probs

    @property
    def mode(self):
        """Mode of geometric distribution."""
        return paddle.zeros_like(self.probs)

    @property
    def variance(self):
        """Variance of geometric distribution"""
        return paddle.to_tensor((self.one_tensor / self.probs - self.one_tensor) / self.probs, dtype=self.probs.dtype)

    @property
    def stddev(self):
        """Standard deviation of Geometric distribution"""
        return paddle.sqrt(self.variance)

    def pmf(self, k):
        """probability mass funciotn evaluated at k

        Args:
            k (int): Value to be evaluated.

        Returns:
            Tensor: Probability.
        """
        if isinstance(k, (numbers.Integral, framework.Variable)):
            return paddle.pow((1 - self.probs), k - 1) * self.probs
        else:
            raise TypeError(f"k must be int|Variable,but get {type(k)}")

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
            raise TypeError(f"k must be int|Variable,but get {type(k)}")

    def sample(self, shape=()):
        """Sample from Geometric distribution with sample shape.

        Args:
            shape (Sequence[int], optional): Sample shape.

        Returns:
            Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`.
        """
        if not isinstance(shape, tuple):
            raise TypeError(f'sample shape must be tuple, but get {type(shape)}.')

        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=()):
        """Generate samples of the specified shape.

        Args:
            shape(tuple): The shape of generated samples.

        Returns:
            Tensor: A sample tensor that fits the Geometric distribution.
        """
        if not isinstance(shape, tuple):
            raise TypeError(f'sample shape must be tuple, but get {type(shape)}.')

        shape = Distribution._extend_shape(self, sample_shape=shape)
        tiny = np.finfo(dtype='float32').tiny

        sample_uniform = Un(low=float(tiny),
                            high=float(1)
                            )

        new_t = sample_uniform.sample(list(shape))
        return (paddle.log(new_t) / paddle.log1p(-(self.probs)))

    def entropy(self):
        """Entropy of dirichlet distribution

        Returns:
            Tensor: Entropy.
        """
        x = (self.one_tensor - self.probs) * paddle.log(self.one_tensor - self.probs)
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
            return 1. - paddle.pow((1. - self.probs), k)
        else:
            raise TypeError(f"k must be int|Variable,but get {type(k)}")

    def kl_divergence(self, other):
        """Calculate the KL divergence KL(self || other) with two Geometric instances.

        Args:
            other (Geometric): An instance of Geometric.

        Returns:
            Tensor: The kl-divergence between two geometric distributions.
        """
        temp = paddle.log(self.probs / other.probs)
        kl_diff = self.probs * paddle.abs(temp)
        return paddle.sum(kl_diff, axis=-1)
