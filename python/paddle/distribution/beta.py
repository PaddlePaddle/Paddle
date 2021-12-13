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

from .dirichlet import Dirichlet
from .exponential_family import ExponentialFamily


class Beta(ExponentialFamily):
    """Beta distribution parameterized by

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
        alpha (float|Tensor): alpha parameter of beta distribution
        beta (float|Tensor): beta parameter of beta distribution


    Examples:
    .. code-block:: python

      import paddle
      from paddle.distribution import Normal

      # Define a single scalar Normal distribution.
      dist = Normal(loc=0., scale=3.)
    """

    def __init__(self, alpha, beta):
        if isinstance(alpha, numbers.Real):
            alpha = paddle.to_tensor(alpha)

        if isinstance(beta, numbers.Real):
            beta = paddle.to_tensor(beta)

        alpha, beta = paddle.broadcast_tensors([alpha, beta])

        self._dirichlet = Dirichlet(paddle.stack([alpha, beta], -1))

        super(Beta, self).__init__(self._dirichlet._batch_shape)

    @property
    def alpha(self):
        """Return alpha parameter of beta distribution.

        Returns:
            alpha parameter
        """
        return self._dirichlet.concentration[..., 0]

    @property
    def beta(self):
        """Return beta parameter of beta distribution.

        Returns:
            beta parameter
        """
        return self._dirichlet.concentration[..., 1]

    @property
    def mean(self):
        """mean of beta distribution.
        """
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self):
        """variance of beat distribution
        """
        sum = self.alpha + self.beta
        return self.alpha * self.beta / (sum.pow(2) * (sum + 1))

    def prob(self, value):
        """probability density funciotn evaluated at value

        Args:
            value (Tensor): value to be evaluated
        """
        return paddle.exp(self.log_prob(value))

    def log_prob(self, value):
        """log probability density funciton evaluated at value

        Args:
            value (Tensor): value to be evaluated
        """
        return self._dirichlet.log_prob(paddle.stack([value, 1.0 - value], -1))

    def sample(self, shape=None):
        """sample from beta distribution with sample shape 

        Args:
            shape (Tensor): sample shape

        Returns:
            sampled data
        """
        return self._dirichlet.sample(shape).select(-1, 0)

    def entropy(self):
        """entropy of dirichlet distribution

        Returns:
            Tensor: [description]
        """
        return self._dirichlet.entropy()
