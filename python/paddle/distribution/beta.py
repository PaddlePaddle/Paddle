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

import paddle
from paddle.distribution import dirichlet, exponential_family


class Beta(exponential_family.ExponentialFamily):
    r"""
    Beta distribution parameterized by alpha and beta.

    In probability theory and statistics, the beta distribution is a family of 
    continuous probability distributions defined on the interval [0, 1] 
    parameterized by two positive shape parameters, denoted by alpha and beta, 
    that appear as exponents of the random variable and control the shape of 
    the distribution. The generalization to multiple variables is called a 
    Dirichlet distribution.

    The probability density function (pdf) is

    .. math::

        f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)}x^{\alpha-1}(1-x)^{\beta-1}

    where the normalization, B, is the beta function,

    .. math::

        B(\alpha, \beta) = \int_{0}^{1} t^{\alpha - 1} (1-t)^{\beta - 1}\mathrm{d}t 


    Args:
        alpha (float|Tensor): Alpha parameter. It supports broadcast semantics. 
            The value of alpha must be positive. When the parameter is a tensor, 
            it represents multiple independent distribution with 
            a batch_shape(refer to ``Distribution`` ).
        beta (float|Tensor): Beta parameter. It supports broadcast semantics. 
            The value of beta must be positive(>0). When the parameter is tensor, 
            it represent multiple independent distribution with 
            a batch_shape(refer to ``Distribution`` ). 

    Examples:

        .. code-block:: python

            import paddle

            # scale input
            beta = paddle.distribution.Beta(alpha=0.5, beta=0.5)
            print(beta.mean)
            # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [0.50000000])
            print(beta.variance)
            # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [0.12500000])
            print(beta.entropy())
            # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [0.12500000])

            # tensor input with broadcast
            beta = paddle.distribution.Beta(alpha=paddle.to_tensor([0.2, 0.4]), beta=0.6)
            print(beta.mean)
            # Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [0.25000000, 0.40000001])
            print(beta.variance)
            # Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [0.10416666, 0.12000000])
            print(beta.entropy())
            # Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        [-1.91923141, -0.38095069])
    """

    def __init__(self, alpha, beta):
        if isinstance(alpha, numbers.Real):
            alpha = paddle.full(shape=[1], fill_value=alpha)

        if isinstance(beta, numbers.Real):
            beta = paddle.full(shape=[1], fill_value=beta)

        self.alpha, self.beta = paddle.broadcast_tensors([alpha, beta])

        self._dirichlet = dirichlet.Dirichlet(
            paddle.stack([self.alpha, self.beta], -1))

        super(Beta, self).__init__(self._dirichlet._batch_shape)

    @property
    def mean(self):
        """Mean of beta distribution.
        """
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self):
        """Variance of beat distribution
        """
        sum = self.alpha + self.beta
        return self.alpha * self.beta / (sum.pow(2) * (sum + 1))

    def prob(self, value):
        """Probability density funciotn evaluated at value

        Args:
            value (Tensor): Value to be evaluated.
        
        Returns:
            Tensor: Probability.
        """
        return paddle.exp(self.log_prob(value))

    def log_prob(self, value):
        """Log probability density funciton evaluated at value

        Args:
            value (Tensor): Value to be evaluated
        
        Returns:
            Tensor: Log probability.
        """
        return self._dirichlet.log_prob(paddle.stack([value, 1.0 - value], -1))

    def sample(self, shape=()):
        """Sample from beta distribution with sample shape.

        Args:
            shape (Sequence[int], optional): Sample shape.

        Returns:
            Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`.
        """
        shape = shape if isinstance(shape, tuple) else tuple(shape)
        return paddle.squeeze(self._dirichlet.sample(shape)[..., 0], axis=-1)

    def entropy(self):
        """Entropy of dirichlet distribution

        Returns:
            Tensor: Entropy.
        """
        return self._dirichlet.entropy()

    @property
    def _natural_parameters(self):
        return (self.alpha, self.beta)

    def _log_normalizer(self, x, y):
        return paddle.lgamma(x) + paddle.lgamma(y) - paddle.lgamma(x + y)
