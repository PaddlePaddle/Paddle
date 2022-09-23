# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distribution.transform import ExpTransform
from paddle.distribution.transformed_distribution import TransformedDistribution
from paddle.distribution.normal import Normal


class LogNormal(TransformedDistribution):
    r"""The Normal distribution with location `loc` and `scale` parameters.

    Mathematical details

    The probability density function (pdf) is

    .. math::
        pdf(x; \mu, \sigma) = \frac{1}{\sigma x \sqrt{2\pi}}e^{(-\frac{(ln(x) - \mu)^2}{2\sigma^2})}

    In the above equation:

    * :math:`loc = \mu`: is the means of the underlying Normal distribution.
    * :math:`scale = \sigma`: is the stddevs of the underlying Normal distribution.

    Args:
        loc(int|float|list|tuple|numpy.ndarray|Tensor): The mean of normal distribution.The data type is int, float, list, numpy.ndarray or Tensor.
        scale(int|float|list|tuple|numpy.ndarray|Tensor): The std of normal distribution.The data type is int, float, list, numpy.ndarray or Tensor.
        name(str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

          import paddle
          from paddle.distribution import LogNormal

          # Define a single scalar LogNormal distribution.
          dist = LogNormal(loc=0., scale=3.)
          # Define a batch of two scalar valued LogNormals.
          # The underlying Normal of first has mean 1 and standard deviation 11, the underlying Normal of second 2 and 22.
          dist = LogNormal(loc=[1., 2.], scale=[11., 22.])
          # Get 3 samples, returning a 3 x 2 tensor.
          dist.sample([3])

          # Define a batch of two scalar valued LogNormals.
          # Their underlying Normal have mean 1, but different standard deviations.
          dist = LogNormal(loc=1., scale=[11., 22.])

          # Complete example
          value_tensor = paddle.to_tensor([0.8], dtype="float32")

          lognormal_a = LogNormal([0.], [1.])
          lognormal_b = LogNormal([0.5], [2.])
          sample = lognormal_a.sample([2])
          # a random tensor created by normal distribution with shape: [2, 1]
          entropy = lognormal_a.entropy()
          # [1.4189385] with shape: [1]
          lp = lognormal_a.log_prob(value_tensor)
          # [-0.72069150] with shape: [1]
          p = lognormal_a.probs(value_tensor)
          # [0.48641577] with shape: [1]
          kl = lognormal_a.kl_divergence(lognormal_b)
          # [0.34939718] with shape: [1]
    """

    def __init__(self, loc, scale, name=None):
        self.base_dist = Normal(loc=loc, scale=scale, name=name)
        self.loc = self.base_dist.loc
        self.scale = self.base_dist.scale
        super(LogNormal, self).__init__(self.base_dist, [ExpTransform()])

    @property
    def mean(self):
        """Mean of lognormal distribuion.

        Returns:
            Tensor: mean value.
        """
        return paddle.exp(self.base_dist.mean + self.base_dist.variance / 2)

    @property
    def variance(self):
        """Variance of lognormal distribution.

        Returns:
            Tensor: variance value.
        """
        return (paddle.expm1(self.base_dist.variance) *
                paddle.exp(2 * self.base_dist.mean + self.base_dist.variance))

    def entropy(self):
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            entropy(\sigma) = 0.5 \\log (2 \pi e \sigma^2) + \mu

        In the above equation:

        * :math:`scale = \sigma`: is the std.

        Returns:
          Tensor: Shannon entropy of lognormal distribution.The data type is float32.

        """
        return self.base_dist.entropy() + self.base_dist.mean

    def probs(self, value):
        """Probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: probability.The data type is same with value.

        """
        return paddle.exp(self.log_prob(value))

    def kl_divergence(self, other):
        r"""The KL-divergence between two lognormal distributions.

        The probability density function (pdf) is

        .. math::

            KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = 0.5 (ratio^2 + (\\frac{diff}{\sigma_1})^2 - 1 - 2 \\ln {ratio})

        .. math::

            ratio = \\frac{\sigma_0}{\sigma_1}

        .. math::

            diff = \mu_1 - \mu_0

        In the above equation:

        * :math:`loc = \mu_0`: is the mean of underlying Normal distribution.
        * :math:`scale = \sigma_0`: is the std of underlying Normal distribution.
        * :math:`loc = \mu_1`: is the mean of other underlying Normal distribution.
        * :math:`scale = \sigma_1`: is the std of other underlying Normal distribution.
        * :math:`ratio`: is the ratio of scales.
        * :math:`diff`: is the difference between means.

        Args:
            other (LogNormal): instance of LogNormal.

        Returns:
            Tensor: kl-divergence between two lognormal distributions.The data type is float32.

        """
        return self.base_dist.kl_divergence(other.base_dist)
