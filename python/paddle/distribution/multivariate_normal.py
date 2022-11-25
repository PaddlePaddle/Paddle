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
import math
from paddle.distribution import distribution


class MultivariateNormal(distribution.Distribution):
    r"""
       (MultivariateNormal Introduce)

       Args:

       Examples:

       """

    def __init__(self, loc, covariance_matrix=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) != 1:
            raise ValueError("Exactly covariance_matrix may be specified.")

        if covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            if(covariance_matrix.shape[:-2] == [] or loc.shape[:-1] == []):
                batch_shape = []
            else:
                batch_shape = paddle.broadcast_shape(covariance_matrix.shape[:-2], loc.shape[:-1])
            self.covariance_matrix = covariance_matrix.expand(batch_shape + [-1, -1])
        self.loc = loc.expand(batch_shape + [-1])

        event_shape = self.loc.shape[-1:]
        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

        if covariance_matrix is not None:
            self._unbroadcasted_scale_tril = paddle.linalg.cholesky(covariance_matrix)

    def covariance_matrix(self):
         res1 = paddle.matmul(self._unbroadcasted_scale_tril,
                             self._unbroadcasted_scale_tril.T)

         return res1.expand(res1, self._batch_shape + self._event_shape + self._event_shape)

    @property
    def mean(self):
        """mean of multivariate_normal distribuion.

        Returns:
            Tensor: mean value.
        """
        return self.loc

    @property
    def variance(self):
        """variance of multivariate_normal distribution.

        Returns:
            Tensor: variance value.
        """
        matrix_decompos = paddle.linalg.cholesky(self.covariance_matrix).pow(2).sum(-1)
        return paddle.broadcast_to(matrix_decompos, self._batch_shape + self._event_shape)

    @property
    def stddev(self):
        """standard deviation of multivariate_normal distribution.

        Returns:
            Tensor: variance value.
        """
        return paddle.sqrt(self.variance)

    def prob(self, value):
        """probability mass function evaluated at value.

        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: probability of value.
        """
        x = paddle.pow(2 * math.pi, -value.shape.pop(1) * 0.5) * paddle.pow(paddle.linalg.det(self.covariance_matrix),
                                                                          -0.5)
        y = paddle.exp(
            -0.5 * paddle.t(value - self.loc) * paddle.inverse(self.covariance_matrix) * (value - self.loc))

        return x * y

    def log_prob(self, value):
        """probability mass function evaluated of logarithm at value

        Args:
            value (Tensor): value to be evaluated.

        Returns:
            Tensor: probability of value.
        """
        return paddle.log(self.prob(value))

    def entropy(self):
        """entropy of multivariate_normal distribution

        Returns:
            Tensor: entropy value
        """
        sigma = paddle.linalg.det(self.covariance_matrix)
        return 0.5 * paddle.log(paddle.pow(2 * math.pi * math.e, self.loc.dim()) * sigma)

    def sample(self, shape=()):
        """draw sample data from multivariate_normal distribution

        Args:
            shape (tuple, optional): [description]. Defaults to ().
        """
        with paddle.no_grad:
            self.rsample(shape)

    def rsample(self, shape=()):
        """draw sample data from multivariate_normal distribution

        Args:
            shape (tuple, optional): [description]. Defaults to ().
        """
        shape = self._extend_shape(shape)
        eps = paddle.standard_normal(shape, dtype=None, name=None)
        unbroadcasted_scale_tril = paddle.linalg.cholesky(self.covariance_matrix)

        return self.loc + self._batch_mv(unbroadcasted_scale_tril, eps)


    def kl_divergence(self, other):
        """calculate the KL divergence KL(self || other) with two MultivariateNormal instances.

        Args:
            other (MultivariateNormal): An instance of MultivariateNormal.

        Returns:
            Tensor: The kl-divergence between two multivariate_normal distributions.
        """
        sector_1 = paddle.t(self.loc - other.loc) * paddle.inverse(other.covariance_matrix) * (self.loc - other.loc)
        sector_2 = paddle.log(paddle.linalg.det(paddle.inverse(other.covariance_matrix) * self.covariance_matrix))
        sector_3 = paddle.trace(paddle.inverse(other.covariance_matrix) * self.covariance_matrix)
        n = self.loc.shape.pop(1)
        return 0.5 * (sector_1 - sector_2 + sector_3 - n)


    def _batch_mv(self,bmat, bvec):
        bvec_unsqueeze = paddle.unsqueeze(bvec, 1)
        bvec = paddle.squeeze(bvec_unsqueeze)
        return paddle.matmul(bmat, bvec)
