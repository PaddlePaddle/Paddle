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

import paddle

from .exponential_family import ExponentialFamily


class Dirichlet(ExponentialFamily):
    """Dirichlet distribution with parameter concentration

    The Dirichlet distribution is defined over the `(k-1)-simplex` using a 
    positive, lenght-k vector concentration(`k > 1`).
    The Dirichlet is identically the Beta distribution when `k = 2`.

    Mathematical details

    The probability density function (pdf) is

    .. math::

        f(x_1,...,x_k; \alpha_1,...,\alpha_k) = \frac{1}{B(\alpha)} \prod_{i=1}^{k}x_i^{\alpha_i-1} 

    The normalizing constant is the multivariate beta function.

    Args:
        concentration (Tensor): concentration parameter of dirichlet 
        distribution

    Examples:
    .. code-block:: python

        import paddle

        dirichlet = paddle.distribution.Dirichlet(paddle.to_tensor([1., 2., 3.]))

        print(dirichlet.entropy())
        # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        #        [-1.24434423])
        print(dirichlet.prob(paddle.to_tensor([.3, .5, .6])))
        # Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        #        [10.80000114])

    """

    def __init__(self, concentration):
        if concentration.dim() < 1:
            raise ValueError(
                "`concentration` parameter must be at least one dimensional")

        self.concentration = concentration
        super(Dirichlet, self).__init__(concentration.shape[:-1],
                                        concentration.shape[-1:])

    @property
    def mean(self):
        """mean of Dirichelt distribution.

        Returns:
            mean value of distribution.
        """
        return self.concentration / self.concentration.sum(-1, keepdim=True)

    @property
    def variance(self):
        """variance of Dirichlet distribution.

        Returns:
            variance value of distribution.
        """
        concentration0 = self.concentration.sum(-1, keepdim=True)
        return (self.concentration * (concentration0 - self.concentration)) / (
            concentration0.pow(2) * (concentration0 + 1))

    def sample(self, shape=None):
        """sample from dirichlet distribution.

        Args:
            shape (Tensor, optional): sample shape. Defaults to None.
        """
        raise NotImplementedError

    def prob(self, value):
        """Probability density function(pdf) evaluated at value.

        Args:
            value (Tensor): value to be evaluated.

        Returns:
            pdf evaluated at value.
        """
        return paddle.exp(self.log_prob(value))

    def log_prob(self, value):
        """log of probability densitiy function.

        Args:
            value (Tensor): value to be evaluated.
        """
        return ((paddle.log(value) * (self.concentration - 1.0)
                 ).sum(-1) + paddle.lgamma(self.concentration.sum(-1)) -
                paddle.lgamma(self.concentration).sum(-1))

    def entropy(self):
        """entropy of Dirichlet distribution.

        Returns:
            entropy of distribution.
        """
        concentration0 = self.concentration.sum(-1)
        k = self.concentration.shape[-1]
        return (paddle.lgamma(self.concentration).sum(-1) -
                paddle.lgamma(concentration0) -
                (k - concentration0) * paddle.digamma(concentration0) - (
                    (self.concentration - 1.0
                     ) * paddle.digamma(self.concentration)).sum(-1))

    @property
    def _natural_parameters(self):
        return (self.concentration, )

    def _log_normalizer(self, x):
        return x.lgamma().sum(-1) - paddle.lgamma(x.sum(-1))
