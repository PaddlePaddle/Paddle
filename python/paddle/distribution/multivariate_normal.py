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

import math
from collections.abc import Sequence

import paddle
from paddle.distribution import distribution


class MultivariateNormal(distribution.Distribution):
    r"""The Multivariate Normal distribution is a type multivariate continuous distribution defined on the real set, with parameter: `loc` and any one
    of the following parameters characterizing the variance: `covariance_matrix`, `precision_matrix`, `scale_tril`.

    Mathematical details

    The probability density function (pdf) is

    .. math::

        p(X ;\mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp(-\frac{1}{2}(X - \mu)^{\intercal} \Sigma^{-1} (X - \mu))

    In the above equation:

    * :math:`X`: is a k-dim random vector.
    * :math:`loc = \mu`: is the k-dim mean vector.
    * :math:`covariance_matrix = \Sigma`: is the k-by-k covariance matrix.

    Args:
        loc(int|float|Tensor): The mean of Multivariate Normal distribution. If the input data type is int or float, the data type of `loc` will be
            convert to a 1-D Tensor the paddle global default dtype.
        covariance_matrix(Tensor): The covariance matrix of Multivariate Normal distribution. The data type of `covariance_matrix` will be convert
            to be the same as the type of loc.
        precision_matrix(Tensor): The inverse of the covariance matrix. The data type of `precision_matrix` will be convert to be the same as the
            type of loc.
        scale_tril(Tensor): The cholesky decomposition (lower triangular matrix) of the covariance matrix. The data type of `scale_tril` will be
            convert to be the same as the type of loc.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import MultivariateNormal
            >>> paddle.set_device("cpu")
            >>> paddle.seed(100)

            >>> rv = MultivariateNormal(loc=paddle.to_tensor([2.,5.]), covariance_matrix=paddle.to_tensor([[2.,1.],[1.,2.]]))

            >>> print(rv.sample([3, 2]))
            Tensor(shape=[3, 2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[-0.00339603,  4.31556797],
              [ 2.01385283,  4.63553190]],
             [[ 0.10132277,  3.11323833],
              [ 2.37435842,  3.56635118]],
             [[ 2.89701366,  5.10602522],
              [-0.46329355,  3.14768648]]])

            >>> print(rv.mean)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [2., 5.])

            >>> print(rv.variance)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [1.99999988, 2.        ])

            >>> print(rv.entropy())
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            3.38718319)

            >>> rv1 = MultivariateNormal(loc=paddle.to_tensor([2.,5.]), covariance_matrix=paddle.to_tensor([[2.,1.],[1.,2.]]))
            >>> rv2 = MultivariateNormal(loc=paddle.to_tensor([-1.,3.]), covariance_matrix=paddle.to_tensor([[3.,2.],[2.,3.]]))
            >>> print(rv1.kl_divergence(rv2))
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.55541301)
    """

    def __init__(
        self,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
    ):
        self.dtype = paddle.get_default_dtype()
        if isinstance(loc, (float, int)):
            loc = paddle.to_tensor([loc], dtype=self.dtype)
        else:
            self.dtype = loc.dtype
        if loc.dim() < 1:
            loc = loc.reshape((1,))
        self.covariance_matrix = None
        self.precision_matrix = None
        self.scale_tril = None
        if (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            scale_tril = paddle.cast(scale_tril, dtype=self.dtype)
            batch_shape = paddle.broadcast_shape(
                scale_tril.shape[:-2], loc.shape[:-1]
            )
            self.scale_tril = scale_tril.expand(
                batch_shape + [scale_tril.shape[-2], scale_tril.shape[-1]]
            )
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            covariance_matrix = paddle.cast(covariance_matrix, dtype=self.dtype)
            batch_shape = paddle.broadcast_shape(
                covariance_matrix.shape[:-2], loc.shape[:-1]
            )
            self.covariance_matrix = covariance_matrix.expand(
                batch_shape
                + [covariance_matrix.shape[-2], covariance_matrix.shape[-1]]
            )
        else:
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            precision_matrix = paddle.cast(precision_matrix, dtype=self.dtype)
            batch_shape = paddle.broadcast_shape(
                precision_matrix.shape[:-2], loc.shape[:-1]
            )
            self.precision_matrix = precision_matrix.expand(
                batch_shape
                + [precision_matrix.shape[-2], precision_matrix.shape[-1]]
            )
        self._check_constraints()
        self.loc = loc.expand(
            batch_shape
            + [
                -1,
            ]
        )
        event_shape = self.loc.shape[-1:]

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = paddle.linalg.cholesky(
                covariance_matrix
            )
        else:
            self._unbroadcasted_scale_tril = precision_to_scale_tril(
                precision_matrix
            )

        super().__init__(batch_shape, event_shape)

    def _check_lower_triangular(self, value):
        """Check whether the input is a lower triangular matrix

        Args:
            value (Tensor): input matrix

        Return:
            Tensor: indicator for lower triangular matrix
        """
        tril = paddle.tril(value)
        is_lower_triangular = paddle.cast(
            (tril == value).reshape(
                value.shape[:-2]
                + [
                    -1,
                ]
            ),
            dtype=self.dtype,
        ).min(-1, keepdim=True)[0]
        is_positive_diag = paddle.cast(
            (value.diagonal(axis1=-2, axis2=-1) > 0), dtype=self.dtype
        ).min(-1, keepdim=True)[0]
        return is_lower_triangular and is_positive_diag

    def _check_positive_definite(self, value):
        """Check whether the input is a positive definite matrix

        Args:
            value (Tensor): input matrix

        Return:
            Tensor: indicator for positive definite matrix
        """
        is_square = paddle.full(
            shape=value.shape[:-2],
            fill_value=(value.shape[-2] == value.shape[-1]),
            dtype="bool",
        ).all()
        if not is_square:
            raise ValueError(
                "covariance_matrix or precision_matrix must be a square matrix"
            )
        new_perm = list(range(len(value.shape)))
        new_perm[-1], new_perm[-2] = new_perm[-2], new_perm[-1]
        is_symmetric = paddle.isclose(
            value, value.transpose(new_perm), atol=1e-6
        ).all()
        if not is_symmetric:
            raise ValueError(
                "covariance_matrix or precision_matrix must be a symmetric matrix"
            )
        is_positive_definite = (
            paddle.cast(paddle.linalg.eigvalsh(value), dtype=self.dtype) > 0
        ).all()
        return is_positive_definite

    def _check_constraints(self):
        """Check whether the matrix satisfy corresponding constraint

        Return:
            Tensor: indicator for the pass of constraints check
        """
        if self.scale_tril is not None:
            check = self._check_lower_triangular(self.scale_tril)
            if not check:
                raise ValueError(
                    "scale_tril matrix must be a lower triangular matrix with positive diagonals"
                )
        elif self.covariance_matrix is not None:
            is_positive_definite = self._check_positive_definite(
                self.covariance_matrix
            )
            if not is_positive_definite:
                raise ValueError(
                    "covariance_matrix must be a symmetric positive definite matrix"
                )
        else:
            is_positive_definite = self._check_positive_definite(
                self.precision_matrix
            )
            if not is_positive_definite:
                raise ValueError(
                    "precision_matrix must be a symmetric positive definite matrix"
                )

    @property
    def mean(self):
        """Mean of Multivariate Normal distribution.

        Returns:
            Tensor: mean value.
        """
        return self.loc

    @property
    def variance(self):
        """Variance of Multivariate Normal distribution.

        Returns:
            Tensor: variance value.
        """
        return (
            paddle.square(self._unbroadcasted_scale_tril)
            .sum(-1)
            .expand(self._batch_shape + self._event_shape)
        )

    def sample(self, shape=()):
        """Generate Multivariate Normal samples of the specified shape. The final shape would be ``sample_shape + batch_shape + event_shape``.

        Args:
            shape (Sequence[int], optional): Prepended shape of the generated samples.

        Returns:
            Tensor, Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`. The data type is the same as `self.loc`.
        """
        with paddle.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=()):
        """Generate Multivariate Normal samples of the specified shape. The final shape would be ``sample_shape + batch_shape + event_shape``.

        Args:
            shape (Sequence[int], optional): Prepended shape of the generated samples.

        Returns:
            Tensor, Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`. The data type is the same as `self.loc`.
        """
        if not isinstance(shape, Sequence):
            raise TypeError('sample shape must be Sequence object.')
        output_shape = self._extend_shape(shape)
        eps = paddle.cast(paddle.normal(shape=output_shape), dtype=self.dtype)
        return self.loc + paddle.matmul(
            self._unbroadcasted_scale_tril, eps.unsqueeze(-1)
        ).squeeze(-1)

    def log_prob(self, value):
        """Log probability density function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability. The data type is the same as `self.loc`.
        """
        value = paddle.cast(value, dtype=self.dtype)

        diff = value - self.loc
        M = batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(axis1=-2, axis2=-1)
            .log()
            .sum(-1)
        )
        return (
            -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M)
            - half_log_det
        )

    def prob(self, value):
        """Probability density function.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor: probability. The data type is the same as `self.loc`.
        """
        return paddle.exp(self.log_prob(value))

    def entropy(self):
        r"""Shannon entropy in nats.

        The entropy is

        .. math::

            \mathcal{H}(X) = \frac{n}{2} \log(2\pi) + \log {\det A} + \frac{n}{2}

        In the above equation:

        * :math:`\Omega`: is the support of the distribution.

        Returns:
            Tensor, Shannon entropy of Multivariate Normal distribution. The data type is the same as `self.loc`.
        """
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(axis1=-2, axis2=-1)
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

    def kl_divergence(self, other):
        r"""The KL-divergence between two poisson distributions with the same `batch_shape` and `event_shape`.

        The probability density function (pdf) is

        .. math::

            KL\_divergence(\lambda_1, \lambda_2) = \log(\det A_2) - \log(\det A_1) -\frac{n}{2} +\frac{1}{2}[tr [\Sigma_2^{-1} \Sigma_1] + (\mu_1 - \mu_2)^{\intercal} \Sigma_2^{-1}  (\mu_1 - \mu_2)]

        Args:
            other (MultivariateNormal): instance of Multivariate Normal.

        Returns:
            Tensor, kl-divergence between two Multivariate Normal distributions. The data type is the same as `self.loc`.

        """
        if (
            self._batch_shape != other._batch_shape
            and self._event_shape != other._event_shape
        ):
            raise ValueError(
                "KL divergence of two Multivariate Normal distributions should share the same `batch_shape` and `event_shape`."
            )
        half_log_det_1 = (
            self._unbroadcasted_scale_tril.diagonal(axis1=-2, axis2=-1)
            .log()
            .sum(-1)
        )
        half_log_det_2 = (
            other._unbroadcasted_scale_tril.diagonal(axis1=-2, axis2=-1)
            .log()
            .sum(-1)
        )
        new_perm = list(range(len(self._unbroadcasted_scale_tril.shape)))
        new_perm[-1], new_perm[-2] = new_perm[-2], new_perm[-1]
        cov_mat_1 = paddle.matmul(
            self._unbroadcasted_scale_tril,
            self._unbroadcasted_scale_tril.transpose(new_perm),
        )
        cov_mat_2 = paddle.matmul(
            other._unbroadcasted_scale_tril,
            other._unbroadcasted_scale_tril.transpose(new_perm),
        )
        expectation = (
            paddle.linalg.solve(cov_mat_2, cov_mat_1)
            .diagonal(axis1=-2, axis2=-1)
            .sum(-1)
        )
        expectation += batch_mahalanobis(
            other._unbroadcasted_scale_tril, self.loc - other.loc
        )
        return (
            half_log_det_2
            - half_log_det_1
            + 0.5 * (expectation - self._event_shape[0])
        )


def precision_to_scale_tril(P):
    """Convert precision matrix to scale tril matrix

    Args:
        P (Tensor): input precision matrix

    Returns:
        Tensor: scale tril matrix
    """
    Lf = paddle.linalg.cholesky(paddle.flip(P, (-2, -1)))
    tmp = paddle.flip(Lf, (-2, -1))
    new_perm = list(range(len(tmp.shape)))
    new_perm[-2], new_perm[-1] = new_perm[-1], new_perm[-2]
    L_inv = paddle.transpose(tmp, new_perm)
    Id = paddle.eye(P.shape[-1], dtype=P.dtype)
    L = paddle.linalg.triangular_solve(L_inv, Id, upper=False)
    return L


def batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance of the Multivariate Normal distribution with cholesky decomposition of the covariance matrix.
    Accepts batches for both bL and bx.

    Args:
        bL (Tensor): scale trial matrix (batched)
        bx (Tensor): difference vector(batched)

    Returns:
        Tensor: squared Mahalanobis distance
    """
    n = bx.shape[-1]
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j, i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.transpose(permute_dims)

    flat_L = bL.reshape((-1, n, n))  # shape = b x n x n
    flat_x = bx.reshape((-1, flat_L.shape[0], n))  # shape = c x b x n
    flat_x_swap = flat_x.transpose((1, 2, 0))  # shape = b x n x c
    M_swap = (
        paddle.linalg.triangular_solve(flat_L, flat_x_swap, upper=False)
        .pow(2)
        .sum(-2)
    )  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.transpose(
        permute_inv_dims
    )  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)
