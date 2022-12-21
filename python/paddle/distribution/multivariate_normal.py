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
        return paddle.expand(matrix_decompos, self._batch_shape + self._event_shape)

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
        return paddle.exp(self.log_prob(value))

    def log_prob(self, value):
        # if self._validate_args:
        #     self._validate_sample(value)
        diff = value - self.loc
        M = self._batch_mahalanobis(self._unbroadcasted_scale_tril, diff)

        half_log_det = paddle.diagonal(self._unbroadcasted_scale_tril,axis1=-2, axis2=-1).log().sum(-1)

        return -0.5 * (self.event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def entropy(self):
        """entropy of multivariate_normal distribution

        Returns:
            Tensor: entropy value
        """
        half_log_det = paddle.diagonal(self._unbroadcasted_scale_tril,axis1=-2, axis2=-1).log().sum(-1)
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)

    def sample(self, shape=[]):
        """draw sample data from multivariate_normal distribution

        Args:
            shape (list, optional): [description]. Defaults to [].
        """
        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=[]):
        """draw sample data from multivariate_normal distribution

        Args:
            shape (list, optional): [description]. Defaults to [].
        """
        shape = self._extend_shape(shape)
        eps = paddle.standard_normal(shape, dtype=None, name=None)
        unbroadcasted_scale_tril = paddle.linalg.cholesky(self.covariance_matrix)

        return self.loc + self._batch_mv(unbroadcasted_scale_tril, eps)


    def kl_divergence(self, other):
        """

        """
        if self.event_shape != other.event_shape:
            raise ValueError("KL-divergence between two Multivariate Normals with\
                              different event shapes cannot be computed")

        half_term1 = (paddle.diagonal(self._unbroadcasted_scale_tril, axis1=-2, axis2=-1).log().sum(-1) -
                      paddle.diagonal(other._unbroadcasted_scale_tril, axis1=-2, axis2=-1).log().sum(-1))
        combined_batch_shape = []
        n = self.event_shape[0]
        self_scale_tril = self._unbroadcasted_scale_tril.expand(combined_batch_shape + [n, n])
        other_scale_tril = other._unbroadcasted_scale_tril.expand(combined_batch_shape + [n, n])
        term2 = self._batch_trace_XXT(paddle.linalg.triangular_solve(self_scale_tril, other_scale_tril, upper=False))
        term3 = self._batch_mahalanobis(self._unbroadcasted_scale_tril, (self.loc - other.loc))
        return half_term1 + 0.5 * (term2 + term3 - n)

    def _batch_trace_XXT(self, bmat):
        """

        """
        n = bmat.shape[-1]
        m = bmat.shape[-2]
        flat_trace = paddle.reshape(bmat, [1, m * n]).pow(2).sum(-1)
        if( bmat.shape[:-2] == []):
            return flat_trace
        else:
            return paddle.reshape(flat_trace, bmat.shape[:-2])

    def _batch_mv(self,bmat, bvec):
        """

        """
        bvec_unsqueeze = paddle.unsqueeze(bvec, 1)
        bvec = paddle.squeeze(bvec_unsqueeze)
        return paddle.matmul(bmat, bvec)

    def _batch_mahalanobis(self, bL, bx):
        """

        """
        n = bx.shape[-1]
        bx_batch_shape = bx.shape[:-1]
        bx_batch_dims = len(bx_batch_shape)
        bL_batch_dims = bL.ndim - 2

        outer_batch_dims = bx_batch_dims - bL_batch_dims
        old_batch_dims = outer_batch_dims + bL_batch_dims
        new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
        bx_new_shape = bx.shape[:outer_batch_dims]

        for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
            bx_new_shape += (sx // sL, sL)
        bx_new_shape += [n]
        bx = paddle.reshape(bx, bx_new_shape)

        permute_dims = (list(range(outer_batch_dims)) +
                        list(range(outer_batch_dims, new_batch_dims, 2)) +
                        list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                        [new_batch_dims])

        bx = paddle.transpose(bx, perm=permute_dims)
        # shape = [b, n, n]
        flat_L = paddle.reshape(bL, [-1, n, n])
        # shape = [c, b, n]
        flat_x = paddle.reshape(bx, [-1, flat_L.shape[0], n])
        # shape = [b, n, c]
        flat_x_swap = paddle.transpose(flat_x, perm=[1, 2, 0])
        # shape = [b, c]
        M_swap = paddle.linalg.triangular_solve(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)
        # shape = [c, b]
        M = M_swap.t()

        if bx.shape[:-1] == []:
            return M.sum()
        else:
            # shape = [..., 1, j, i, 1]
            permuted_M = paddle.reshape(M, bx.shape[:-1])
            permute_inv_dims = list(range(outer_batch_dims))

            for i in range(bL_batch_dims):
                permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
                # shape = [..., 1, i, j, 1]
            reshaped_M = paddle.transpose(permuted_M, perm=permute_inv_dims)
            return paddle.reshape(reshaped_M, bx_batch_shape)

    def _extend_shape(self, sample_shape):
        """compute shape of the sample

        Args:
            sample_shape (Tensor): sample shape

        Returns:
            Tensor: generated sample data shape
        """
        return sample_shape + list(self.batch_shape) + list(self.event_shape)
