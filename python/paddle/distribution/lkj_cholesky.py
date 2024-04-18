# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distribution.beta import Beta

__all__ = ["LKJCholesky"]


def mvlgamma(a, p):
    """
    Computes the multivariate gamma function for each element in the input tensor `a` with dimension `p`.
    The multivariate gamma function is an extension of the gamma function to multiple dimensions.
    Args:
        a (paddle.Tensor): A scalar or tensor of shape (...,), representing the input values for the
                           multivariate gamma function.
        p (int): An integer representing the dimension of the multivariate gamma function.

    Returns:
        paddle.Tensor: A tensor with the same shape as the input tensor `a`, containing the result of the
                       multivariate gamma function for each element in `a`.
    """
    p_float = float(p)
    order = paddle.arange(0, p_float, dtype=a.dtype)
    return paddle.sum(paddle.lgamma(a.unsqueeze(-1) - 0.5 * order), axis=-1)


def tril_indices(n, k=0):
    """
    Returns the indices of the lower triangular part of an n x n matrix, including the k-th diagonal.
    Args:
        n (int): The size of the square matrix (n x n).
        k (int, optional): The diagonal to include in the lower triangular part. Default is 0, which
                           corresponds to the main diagonal.
    Returns:
        tuple: A tuple containing two 1D tensors, one for the row indices and one for the column indices
               of the non-zero elements in the lower triangular matrix.
    """
    full_matrix = paddle.ones((n, n), dtype='int32')
    tril_matrix = paddle.tril(full_matrix, diagonal=k)
    rows, cols = paddle.nonzero(tril_matrix, as_tuple=True)
    return rows.flatten(), cols.flatten()


def matrix_to_tril(x, diagonal=0):
    """
    Extracts the lower triangular part of the input matrix or batch of matrices `x`, including the specified diagonal.
    Args:
        x (paddle.Tensor): A square matrix or a batch of square matrices of shape (..., n, n), where n is the matrix size.
        diagonal (int, optional): The diagonal to include in the lower triangular part. Default is 0, which corresponds
                                  to the main diagonal.
    Returns:
        paddle.Tensor: A 1D tensor or a batch of 1D tensors containing the elements of the lower triangular parts of the
                       input matrix or matrices `x`, including the specified diagonal.
    """
    matrix_dim = x.shape[-1]
    rows, cols = tril_indices(matrix_dim, diagonal)
    return x[..., rows, cols]


def construct_matrix_lower(p):
    """
    Constructs a lower triangular matrix from a 1D tensor `p` containing the elements of the lower triangular part.
    Args:
        p (paddle.Tensor): A 1D tensor containing the elements of the lower triangular part, including the main diagonal.
                           Its length must be a triangular number (i.e., 1, 3, 6, 10, ...).
    Returns:
        paddle.Tensor: A square lower triangular matrix of shape (n, n), where n is the matrix size, with its elements
                       filled from the input tensor `p`.
    """
    dim = int((math.sqrt(paddle.to_tensor(1 + 8 * p.shape[0])) + 1) / 2)
    matrix = paddle.zeros(shape=[dim, dim], dtype='float32')

    rows, cols = paddle.meshgrid(paddle.arange(dim), paddle.arange(dim))
    mask = rows > cols

    lower_indices = paddle.stack([rows[mask], cols[mask]], axis=1)
    matrix = paddle.scatter_nd_add(matrix, lower_indices, paddle.flatten(p))

    return matrix


class LKJCholesky(distribution.Distribution):
    """
    The LKJCholesky class represents the LKJ distribution over Cholesky factors of correlation matrices.

    This class implements the LKJ distribution over Cholesky factors of correlation matrices, as described in
    Lewandowski, Kurowicka, and Joe (2009)[1]. It supports two sampling methods: "onion" and "cvine".

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method`,
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe

    Args:
        dim (int): The dimension of the correlation matrices.
        concentration (float, optional): The concentration parameter of the LKJ distribution. Default is 1.0.
        sample_method (str, optional): The sampling method to use, either "onion" or "cvine". Default is "onion".

    Example:

        >>> dim = 3
        >>> lkj = LKJCholesky(dim=dim)
        >>> lkj.sample()
        Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               [[ 1.        ,  0.        ,  0.        ],
                [ 0.23500462,  0.97199428,  0.        ],
                [-0.03465778, -0.92403257,  0.38073963]])

        >>> dim = 3
        >>> sample_method = 'cvine'
        >>> lkj = LKJCholesky(dim = dim, sample_method=sample_method)
        >>> lkj.sample()
        Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                [[1.        , 0.        , 0.        ],
                [0.13515258, 0.99082482, 0.        ],
                [0.12862849, 0.01884033, 0.99151385]])

    """

    def __init__(self, dim, concentration=1.0, sample_method="onion"):
        if dim < 2:
            raise ValueError(
                f"Expected dim to be an integer greater than or equal to 2. Found dim={dim}."
            )
        self.dim = dim
        self.concentration = paddle.to_tensor(concentration)
        self.sample_method = sample_method

        batch_shape = self.concentration.shape
        event_shape = (dim, dim)

        # This is used to draw vectorized samples from the beta distribution in Sec. 3.2 of [1].
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        offset = paddle.arange(
            self.dim - 1,
            dtype=self.concentration.dtype,
        )

        if sample_method == "onion":
            offset = paddle.concat(
                [paddle.zeros((1,), dtype=offset.dtype), offset]
            )
            beta_conc1 = offset + 0.5
            beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
            self._beta = Beta(beta_conc1, beta_conc0)
        elif sample_method == "cvine":
            offset_tril = matrix_to_tril(
                paddle.broadcast_to(0.5 * offset, [self.dim - 1, self.dim - 1])
            )
            beta_conc = marginal_conc.unsqueeze(-1) - offset_tril
            self._beta = Beta(beta_conc, beta_conc)
        else:
            raise ValueError("`method` should be one of 'cvine' or 'onion'.")
        super().__init__(batch_shape, event_shape)

    def _onion(self, sample_shape):
        """Generate a sample using the "onion" method.

        Args:
            sample_shape (tuple): The shape of the samples to be generated.

        Returns:
            w (paddle.Tensor): The Cholesky factor of the sampled correlation matrix.
        """

        # Sample y from the Beta distribution
        y = self._beta.sample(sample_shape).unsqueeze(-1)

        # Sample u from the standard normal distribution and create a lower triangular matrix
        u_normal = paddle.randn(
            self._extend_shape(sample_shape), dtype=y.dtype
        ).tril(-1)

        # Normalize u to get u_hypersphere
        u_hypersphere = u_normal / u_normal.norm(axis=-1, keepdim=True)
        # Replace NaNs in first row
        u_hypersphere[..., 0, :].fill_(0.0)
        w = paddle.sqrt(y) * u_hypersphere

        # Fill diagonal elements; clamp for numerical stability
        eps = paddle.finfo(w.dtype).tiny
        diag_elems = paddle.clip(
            1 - paddle.sum(w**2, axis=-1), min=eps
        ).sqrt()

        w += paddle.diag_embed(diag_elems)
        return w

    def _cvine(self, sample_shape):
        """
        Generate a sample using the "cvine" method.

        Args:
            sample_shape (tuple): The shape of the samples to be generated.

        Returns:
            r (paddle.Tensor): The Cholesky factor of the sampled correlation matrix.
        """
        # Sample beta and calculate partial correlations
        beta_sample = self._beta.sample(sample_shape).unsqueeze(-1)
        partial_correlation = 2 * beta_sample - 1

        # Construct the lower triangular matrix from the partial correlations
        partial_correlation = construct_matrix_lower(partial_correlation)

        # Clip partial correlations for numerical stability
        eps = paddle.finfo(beta_sample.dtype).tiny
        r = paddle.clip(partial_correlation, min=(-1 + eps), max=(1 - eps))

        # Calculate the cumulative product of the square root of 1 - z
        z = r**2
        z1m_cumprod_sqrt = paddle.cumprod(paddle.sqrt(1 - z), dim=-1)

        # Shift the elements and pad with 1.0
        pad_width = [0, 0] * (z1m_cumprod_sqrt.ndim - 1) + [1, 0]
        z1m_cumprod_sqrt_shifted = paddle.nn.functional.pad(
            z1m_cumprod_sqrt[..., :-1],
            pad=pad_width,
            mode="constant",
            value=1.0,
        )

        # Calculate the final Cholesky factor
        r += paddle.eye(
            partial_correlation.shape[-2], partial_correlation.shape[-1]
        )
        return r * z1m_cumprod_sqrt_shifted

    def sample(self, sample_shape=None):
        """Generate a sample using the specified sampling method."""
        if sample_shape is None:
            sample_shape = paddle.to_tensor([])
        if self.sample_method == "onion":
            return self._onion(sample_shape)
        else:
            return self._cvine(sample_shape)

    def log_prob(self, value):
        r"""
        Compute the log probability density of the given Cholesky factor under the LKJ distribution.

        Note about computing Jacobian of the transformation from Cholesky factor to
        correlation matrix:

        Assume C = L@Lt and L = (1 0 0; a \sqrt(1-a^2) 0; b c \sqrt(1-b^2-c^2)), we have
        Then off-diagonal lower triangular vector of L is transformed to the off-diagonal
        lower triangular vector of C by the transform:
            (a, b, c) -> (a, b, ab + c\sqrt(1-a^2))
        Hence, Jacobian = 1 * 1 * \sqrt(1 - a^2) = \sqrt(1 - a^2) = L22, where L22
            is the 2th diagonal element of L
        Generally, for a D dimensional matrix, we have:
            Jacobian = L22^(D-2) * L33^(D-3) * ... * Ldd^0

        From [1], we know that probability of a correlation matrix is propotional to
        determinant ** (concentration - 1) = prod(L_ii ^ 2(concentration - 1))
        On the other hand, Jabobian of the transformation from Cholesky factor to
        correlation matrix is:
        prod(L_ii ^ (D - i))
        So the probability of a Cholesky factor is propotional to
        prod(L_ii ^ (2 * concentration - 2 + D - i)) =: prod(L_ii ^ order_i)
        with order_i = 2 * concentration - 2 + D - i,
        i = 2..D (we omit the element i = 1 because L_11 = 1)

        Args:
            value (paddle.Tensor): The Cholesky factor of the correlation matrix for which the log probability density is to be computed.

        Returns:
            log_prob (paddle.Tensor): The log probability density of the given Cholesky factor under the LKJ distribution.
        """
        # 1.Compute the order vector.
        diag_elems = paddle.diagonal(value, offset=0, axis1=-1, axis2=-2)[
            ..., 1:
        ]
        order = paddle.arange(2, self.dim + 1, dtype=self.concentration.dtype)
        order = 2 * (self.concentration - 1).unsqueeze(-1) + self.dim - order

        # 2.Compute the unnormalized log probability density
        unnormalized_log_pdf = paddle.sum(
            order * paddle.log(diag_elems), axis=-1
        )

        # 3.Compute the normalization constant (page 1999 of [1])
        dm1 = self.dim - 1
        alpha = self.concentration + 0.5 * dm1
        denominator = paddle.lgamma(alpha) * dm1
        numerator = mvlgamma(alpha - 0.5, dm1)

        # 4.Compute the constant term related to pi
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * dm1 * math.log(math.pi)

        # 5.Compute the normalization term and return the final log probability density:
        normalize_term = pi_constant + numerator - denominator
        return unnormalized_log_pdf - normalize_term
