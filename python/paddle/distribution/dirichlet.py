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

    Mathematical details

    The probability density function (pdf) is

    .. math::

        pdf(x; \mu, \sigma) = \\frac{1}{Z}e^{\\frac {-0.5 (x - \mu)^2}  {\sigma^2} }

    .. math::

        Z = (2 \pi \sigma^2)^{0.5}

    In the above equation:

    * :math:`loc = \mu`: is the mean.
    * :math:`scale = \sigma`: is the std.
    * :math:`Z`: is the normalization constant.

    Args:
        concentration (Tensor): concentration parameter of dirichlet 
        distribution
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
        """[summary]

        Args:
            shape ([type], optional): [description]. Defaults to None.
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
        """log of probability densitiy function

        Args:
            value ([type]): [description]
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
