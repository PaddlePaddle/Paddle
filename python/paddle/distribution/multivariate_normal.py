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


from paddle.distribution import distribution

class MultivariateNormal(distribution.Distribution):
    r"""
       (MultivariateNormal Introduce)

       Args:

       Examples:

       """

    def __init__(self):
        pass

    @property
    def mean(self):
        """mean of multivariate_normal distribuion.

        Returns:
            Tensor: mean value.
        """
        pass

    @property
    def variance(self):
        """variance of multivariate_normal distribution.

        Returns:
            Tensor: variance value.
        """
        pass

    @property
    def stddev(self):
        """standard deviation of multivariate_normal distribution.

        Returns:
            Tensor: variance value.
        """
        pass

    def prob(self, value):
        """probability mass function evaluated at value.

        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: probability of value.
        """
        pass

    def log_prob(self, value):
        """probability mass function evaluated of logarithm at value

        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: probability of value.
        """
        pass

    def entropy(self):
        """entropy of multivariate_normal distribution

        Returns:
            Tensor: entropy value
        """
        pass

    def sample(self, shape=()):
        """draw sample data from multivariate_normal distribution

        Args:
            shape (tuple, optional): [description]. Defaults to ().
        """
        pass

    def rsample(self, shape=()):
        """draw sample data from multivariate_normal distribution

        Args:
            shape (tuple, optional): [description]. Defaults to ().
        """
        pass

    def kl_divergence(self, other):
        """calculate the KL divergence KL(self || other) with two MultivariateNormal instances.

        Args:
            other (MultivariateNormal): An instance of MultivariateNormal.

        Returns:
            Tensor: The kl-divergence between two multivariate_normal distributions.
        """
        pass
