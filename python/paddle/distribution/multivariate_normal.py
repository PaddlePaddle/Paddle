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

import math

import paddle
from paddle.distribution import distribution


class MultivariateNormal(distribution.Distribution):
    r"""
    MultivariateNormal distribution parameterized by :attr:`loc` and :attr:`covariance_matrix`.

    The probability mass function (PMF) for multivariate_normal is

    .. math::

        f_\boldsymbol{X}(x_1,...,x_k) = \frac{exp(-\frac{1}{2}$\mathbf{(\boldsymbol{x - \mu})}^\top$\boldsymbol{\Sigma}^{-1}(\boldsymbol{x - \mu}))}{\sqrt{(2\pi)^k\left| \boldsymbol{\Sigma} \right|}}

    In the above equation:

    * :math:`loc = \mu`: is the location parameter.
    * :math:`covariance\_matrix = \Sigma`: is the multivariate normal distribution covariance matrix is established when the covariance matrix is a positive semi-definite matrix.

    Args:
        loc(tensor): MultivariateNormal distribution location parameter. The data type is Tensor.
        covariance\_matrix(tensor): MultivariateNormal distribution covariance matrix parameter. The data type is Tensor, and the parameter must be a positive semi-definite matrix.

    Examples:
        .. code-block:: python

          import paddle
          from paddle.distribution.multivariate_normal import MultivariateNormal
          # MultivariateNormal distributed with loc=torch.tensor([0,1],dtype=torch.float32), covariance_matrix=torch.tensor([[2,1],[1,2]],dtype=torch.float32)
          dist = MultivariateNormal(torch.tensor([0,1],dtype=torch.float32),torch.tensor([[2,1],[1,2]],dtype=torch.float32))
          dist.sample([2,2])
          #Tensor(shape=[2, 2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
          #       [[[-1.24544513,  0.24218500],
          #         [-0.26033771,  0.36445701]],

          #        [[ 0.41002670,  1.30887973],
          #         [-0.39297765,  1.32064724]]])
          value = paddle.to_tensor([[2,1],[1,2]],dtype=paddle.float32)
          dist.prob(value)
          #Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,[0.02422146, 0.06584076])
          dist.log_prob(value)
          #Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,[-3.72051620, -2.72051620])
          dist.entropy()
          #Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,[3.38718319])
          dist.rsample([2,2])
          #Tensor(shape=[2, 2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
          #       [[[-2.64208245,  2.58928585],
          #         [-2.26590896,  2.81269646]],

          #        [[ 1.51346231,  1.07011509],
          #         [ 2.11932302,  0.55175352]]])
          dist_kl = MultivariateNormal(paddle.to_tensor([1,2],dtype=paddle.float32),paddle.to_tensor([[4,2],[2,4]],dtype=paddle.float32))
          dist.kl_divergence(dist_kl)
          #Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,[0.64018595])
    """

    def __init__(self, loc, covariance_matrix):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) != 1:
            raise ValueError("Exactly covariance_matrix may be specified.")

        if covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            if covariance_matrix.shape[:-2] == [] or loc.shape[:-1] == []:
                batch_shape = []
            else:
                batch_shape = paddle.broadcast_shape(
                    covariance_matrix.shape[:-2], loc.shape[:-1]
                )
            self.covariance_matrix = covariance_matrix.expand(
                batch_shape + [-1, -1]
            )

        self.loc = loc.expand(batch_shape + [-1])
        event_shape = self.loc.shape[-1:]
        super(MultivariateNormal, self).__init__(batch_shape, event_shape)

        if covariance_matrix is not None:
            self._unbroadcasted_scale_tril = paddle.linalg.cholesky(
                covariance_matrix
            )

    @property
    def mean(self):
        r"""Mean of distribution

        The mean is

        .. math::

            mean = \mu

        In the above equation:

        * :math:`loc = \mu`: is the location parameter.

        Returns:
            Tensor: mean value.

        """
        return self.loc

    @property
    def variance(self):
        r"""Variance of distribution.

        The variance is

        .. math::

            variance = \boldsymbol{\sigma^2}

        In the above equation:

        * :math:`scale = \sigma`: is scale vector obtained after matrix decomposition of multivariate normal distribution covariance matrix.

        Returns:
            Tensor: The variance value.

        """
        matrix_decompos = (
            paddle.linalg.cholesky(self.covariance_matrix).pow(2).sum(-1)
        )
        return paddle.expand(
            matrix_decompos, self._batch_shape + self._event_shape
        )

    @property
    def stddev(self):
        r"""Standard deviation of distribution

        The standard deviation is

        .. math::

            stddev = \boldsymbol{\sigma}

        In the above equation:
        * :math:`scale = \sigma`: is scale vector obtained after matrix decomposition of multivariate normal distribution covariance matrix.

        Returns:
            Tensor: std value
        """
        return paddle.sqrt(self.variance)

    def prob(self, value):
        r"""Probability density/mass function

        The probability density is

        .. math::

             prob(value) = \frac{exp(-\frac{1}{2}$\mathbf{(\boldsymbol{value - \mu})}^\top$\boldsymbol{\Sigma}^{-1}(\boldsymbol{value- \mu}))}{\sqrt{(2\pi)^k\left| \boldsymbol{\Sigma} \right|}}

        In the above equation:

        * :math:`loc = \mu`: is the location parameter.
        * :math:`covariance\_matrix = \Sigma`: is the multivariate normal distribution covariance matrix is established when the covariance matrix is a positive semi-definite matrix.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: probability.The data type is same with value.
        """
        if not isinstance(value, type(self.loc)):
            raise TypeError(
                f"Expected type of value is {type(self.loc)}, but got {type(value)}"
            )

        return paddle.exp(self.log_prob(value))

    def log_prob(self, value):
        r"""Log probability density/mass function.

        The log probability density is

         .. math::

             log\_prob(value) = log(\frac{exp(-\frac{1}{2}$\mathbf{(\boldsymbol{value - \mu})}^\top$\boldsymbol{\Sigma}^{-1}(\boldsymbol{value- \mu}))}{\sqrt{(2\pi)^k\left| \boldsymbol{\Sigma} \right|}})

        In the above equation:

        * :math:`loc = \mu`: is the location parameter.
        * :math:`covariance\_matrix = \Sigma`: is the multivariate normal distribution covariance matrix is established when the covariance matrix is a positive semi-definite matrix.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: log probability.The data type is same with value.
        """
        if not isinstance(value, type(self.loc)):
            raise TypeError(
                f"Expected type of value is {type(self.loc)}, but got {type(value)}"
            )

        diff = value - self.loc
        M = self._batch_mahalanobis(self._unbroadcasted_scale_tril, diff)

        half_log_det = (
            paddle.diagonal(self._unbroadcasted_scale_tril, axis1=-2, axis2=-1)
            .log()
            .sum(-1)
        )

        return (
            -0.5 * (self.event_shape[0] * math.log(2 * math.pi) + M)
            - half_log_det
        )

    def entropy(self):
        r"""Entropy of multivariate_normal distribution

        The entropy is

        .. math::

            entropy() = \frac{k}{2}(\ln 2\pi + 1) + \frac{1}{2}\ln \left| \boldsymbol{\Sigma} \right|

        In the above equation:

        * :math:`k`: The dimension of the multivariate normal distribution vector, such as one-dimensional vector k=1, two-dimensional vector (matrix) k=2.
        * :math:`covariance\_matrix = \Sigma`: is the multivariate normal distribution covariance matrix is established when the covariance matrix is a positive semi-definite matrix.

        Returns:
            Tensor: entropy value
        """
        half_log_det = (
            paddle.diagonal(self._unbroadcasted_scale_tril, axis1=-2, axis2=-1)
            .log()
            .sum(-1)
        )
        H = (
            0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi))
            + half_log_det
        )
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)

    def sample(self, shape=[]):
        """Draw sample data from multinomial distribution

        Args:
            shape (Sequence[int], optional): Shape of the generated samples. Defaults to [].
        """
        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=[]):
        """Generate reparameterized samples of the specified shape.

        Args:
          shape (Sequence[int], optional): Shape of the generated samples. Defaults to [].

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.

        """
        shape = self._extend_shape(shape)
        eps = paddle.standard_normal(shape, dtype=None, name=None)
        unbroadcasted_scale_tril = paddle.linalg.cholesky(
            self.covariance_matrix
        )

        return self.loc + self._batch_product_mv(unbroadcasted_scale_tril, eps)

    def kl_divergence(self, other):
        r"""Calculate the KL divergence KL(self || other) with two MultivariateNormal instances.

        The kl_divergence between two MultivariateNormal distribution is

        .. math::
            KL\_divergence(\boldsymbol{\mu_1}, \boldsymbol{\Sigma_1}; \boldsymbol{\mu_2}, \boldsymbol{\Sigma_2}) =
            \frac{1}{2}\Big \{\log ratio -n + tr(\boldsymbol{\Sigma_2}^{-1}\boldsymbol{\Sigma_1}) +
            $\mathbf{(diff)}^\top$\boldsymbol{\Sigma_2}^{-1}\boldsymbol{(diff)} \Big \}

        .. math::
            ratio = \frac{\left| \boldsymbol{\Sigma_2} \right|}{\left| \boldsymbol{\Sigma_1} \right|}

        .. math::
            \boldsymbol{diff} = \boldsymbol{\mu_2} - \boldsymbol{\mu_1}

        In the above equation:

        * :math:`loc = \mu_1`: is the location parameter of self.
        * :math:`covariance_matrix = \Sigma_1`: is the covariance_matrix parameter of self.
        * :math:`loc = \mu_2`: is the location parameter of the reference MultivariateNormal distribution.
        * :math:`covariance_matrix = \Sigma_2`: is the covariance_matrix parameter of the reference MultivariateNormal distribution.
        * :math:`ratio`: is the ratio of the determinant values of the two covariance matrices.
        * :math:`diff`: is the difference between the two distribution.
        * :math:`n`: is dimension.
        * :math:`tr`: is matrix trace.

        Args:
            other (MultivariateNormal): instance of MultivariateNormal.

        Returns:
            Tensor: kl-divergence between two multivariate_normal distributions.

        """
        if self.event_shape != other.event_shape:
            raise ValueError(
                "KL-divergence between two Multivariate Normals with\
                              different event shapes cannot be computed"
            )

        sector1 = paddle.diagonal(
            self._unbroadcasted_scale_tril, axis1=-2, axis2=-1
        ).log().sum(-1) - paddle.diagonal(
            other._unbroadcasted_scale_tril, axis1=-2, axis2=-1
        ).log().sum(
            -1
        )
        if list(self.batch_shape) == [] and list(other.batch_shape) == []:
            combined_batch_shape = []
        else:
            combined_batch_shape = [self.batch_shape, other.batch_shape]
        n = self.event_shape[0]
        self_scale_tril = self._unbroadcasted_scale_tril.expand(
            combined_batch_shape + [n, n]
        )
        other_scale_tril = other._unbroadcasted_scale_tril.expand(
            combined_batch_shape + [n, n]
        )
        sector2 = self._batch_trace_XXT(
            paddle.linalg.triangular_solve(
                self_scale_tril, other_scale_tril, upper=False
            )
        )
        sector3 = self._batch_mahalanobis(
            self._unbroadcasted_scale_tril, (self.loc - other.loc)
        )
        return sector1 + 0.5 * (sector2 + sector3 - n)

    def _batch_trace_XXT(self, batch_matrix):
        """Calculate the trace of XX^{T} with X having arbitrary trailing batch dimensions.

        Args:
            batch_matrix (Tensor): a tensor with arbitrary trailing batch dimensions

        Returns:
            Tensor: generated the trace of XX^{T} with X
        """
        n = batch_matrix.shape[-1]
        m = batch_matrix.shape[-2]
        flat_trace = paddle.reshape(batch_matrix, [1, m * n]).pow(2).sum(-1)
        if batch_matrix.shape[:-2] == []:
            return flat_trace
        else:
            return paddle.reshape(flat_trace, batch_matrix.shape[:-2])

    def _batch_product_mv(self, batch_matrix, batch_vector):
        """Performs a batched matrix-vector product, with compatible but different batch shapes.
        Both `batch_matrix` and `batch_vector` may have any number of leading dimensions, which
        correspond to a batch shape. They are not necessarily assumed to have the same batch
        shape,just ones which can be broadcasted.

        Args:
            batch_matrix (Tensor): batch matrix tensor with any number of leading dimensions
            batch_vector (Tensor): batch vector tensor with any number of leading dimensions

        Returns:
            Tensor: a batched matrix-vector product
        """
        batch_vector_unsqueeze = paddle.unsqueeze(batch_vector, 1)
        batch_vector = paddle.squeeze(batch_vector_unsqueeze)
        return paddle.matmul(batch_matrix, batch_vector)

    def _batch_mahalanobis(self, batch_L, batch_x):
        """Computes the squared Mahalanobis distance which assess the similarity between data.
        Accepts batches for both batch_L and batch_x. They are not necessarily assumed to have
        the same batch shape, but `batch_L` one should be able to broadcasted to `batch_x` one.

        Args:
            batch_L (Tensor): tensor after matrix factorization
            batch_x (Tensor): difference between two tensors

        Returns:
            Tensor: the squared Mahalanobis distance
        """
        n = batch_x.shape[-1]
        bx_batch_shape = batch_x.shape[:-1]
        bx_batch_dims = len(bx_batch_shape)
        bL_batch_dims = batch_L.ndim - 2

        outer_batch_dims = bx_batch_dims - bL_batch_dims
        old_batch_dims = outer_batch_dims + bL_batch_dims
        new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
        bx_new_shape = batch_x.shape[:outer_batch_dims]

        for (sL, sx) in zip(
            batch_L.shape[:-2], batch_x.shape[outer_batch_dims:-1]
        ):
            bx_new_shape += (sx // sL, sL)
        bx_new_shape += [n]
        batch_x = paddle.reshape(batch_x, bx_new_shape)

        permute_dims = (
            list(range(outer_batch_dims))
            + list(range(outer_batch_dims, new_batch_dims, 2))
            + list(range(outer_batch_dims + 1, new_batch_dims, 2))
            + [new_batch_dims]
        )

        batch_x = paddle.transpose(batch_x, perm=permute_dims)
        # shape = [b, n, n]
        flat_L = paddle.reshape(batch_L, [-1, n, n])
        # shape = [c, b, n]
        flat_x = paddle.reshape(batch_x, [-1, flat_L.shape[0], n])
        # shape = [b, n, c]
        flat_x_swap = paddle.transpose(flat_x, perm=[1, 2, 0])
        # shape = [b, c]
        M_swap = (
            paddle.linalg.triangular_solve(flat_L, flat_x_swap, upper=False)
            .pow(2)
            .sum(-2)
        )
        # shape = [c, b]
        M = M_swap.t()

        if batch_x.shape[:-1] == []:
            return M.sum()
        else:
            # shape = [..., 1, j, i, 1]
            permuted_M = paddle.reshape(M, batch_x.shape[:-1])
            permute_inv_dims = list(range(outer_batch_dims))

            for i in range(bL_batch_dims):
                permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
            # shape = [..., 1, i, j, 1]
            reshaped_M = paddle.transpose(permuted_M, perm=permute_inv_dims)
            return paddle.reshape(reshaped_M, bx_batch_shape)

    def _extend_shape(self, sample_shape):
        """Compute shape of the sample

        Args:
            sample_shape (Tensor): sample shape

        Returns:
            Tensor: generated sample data shape
        """
        return sample_shape + list(self.batch_shape) + list(self.event_shape)
