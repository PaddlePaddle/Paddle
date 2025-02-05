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
from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from functools import reduce
from typing import TYPE_CHECKING, Literal

import paddle
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.framework import Variable
from paddle.distribution import distribution
from paddle.distribution.beta import Beta
from paddle.framework import in_dynamic_mode

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle._typing.dtype_like import _DTypeLiteral


__all__ = ["LKJCholesky"]


def mvlgamma(a, p):
    """
    Computes the multivariate log gamma function for input `a` and dimension `p`.
    """
    pi = paddle.to_tensor(math.pi, dtype=a.dtype)
    j = paddle.arange(1, p + 1, dtype=a.dtype)
    gammaln_terms = paddle.lgamma(a.unsqueeze(-1) + (1 - j) / 2)
    gammaln_sum = paddle.sum(gammaln_terms, axis=-1)
    return (p * (p - 1) / 4) * paddle.log(pi) + gammaln_sum


def tril_indices(n, k=0):
    """
    Returns the indices of the lower triangular part of an n x n matrix, including the k-th diagonal.
    """
    full_matrix = paddle.ones((n, n), dtype='int32')
    tril_matrix = paddle.tril(full_matrix, diagonal=k)
    rows, cols = paddle.nonzero(tril_matrix, as_tuple=True)
    return rows.flatten(), cols.flatten()


def matrix_to_tril(x, diagonal=0):
    """
    Extracts the lower triangular part of the input matrix or batch of matrices `x`, including the specified diagonal.
    """
    tril_mask = paddle.tril(paddle.ones_like(x), diagonal=diagonal)
    tril_elements = paddle.masked_select(x, tril_mask.astype('bool'))
    return tril_elements


def vec_to_tril_matrix(
    p_flatten, dim, last_dim, flatten_shape, sample_shape=(), diag=0
):
    """
    Constructs a batch of lower triangular matrices from a given input tensor `p`.
    """
    # Calculate the dimension of the square matrix based on the last but one dimension of `p`
    # Define the output shape, which adds two dimensions for the square matrix
    shape0 = flatten_shape // last_dim
    output_shape = (
        *sample_shape,
        shape0 // reduce(operator.mul, sample_shape, 1),
        dim,
        dim,
    )

    # Create index_matrix = [index0, rows, cols]
    rows, cols = paddle.meshgrid(paddle.arange(dim), paddle.arange(dim))
    mask = rows > cols
    lower_indices = paddle.stack([rows[mask], cols[mask]], axis=1)
    repeated_lower_indices = paddle.repeat_interleave(
        lower_indices, shape0, axis=0
    )
    index0 = paddle.arange(shape0).unsqueeze(1).tile([last_dim, 1])
    index_matrix = paddle.concat([index0, repeated_lower_indices], axis=1)

    # Sort the indices
    sorted_indices = paddle.argsort(index_matrix[:, 0])
    index_matrix = index_matrix[sorted_indices]

    # Set the value
    matrix = paddle.zeros(shape=(shape0, dim, dim), dtype=p_flatten.dtype)
    matrix = paddle.scatter_nd_add(matrix, index_matrix, p_flatten).reshape(
        output_shape
    )

    return matrix


def tril_matrix_to_vec(mat: Tensor, diag: int = 0) -> Tensor:
    r"""
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector
    which comprises of lower triangular elements from the matrix in row order.
    """
    out_shape = mat.shape[:-2]
    n = mat.shape[-1]
    if diag < -n or diag >= n:
        raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n-1}].")

    rows, cols = paddle.meshgrid(paddle.arange(n), paddle.arange(n))
    tril_mask = diag + rows >= cols

    vec_len = (n + diag) * (n + diag + 1) // 2
    out_shape += (vec_len,)

    # Use the mask to index the lower triangular elements from the input matrix
    tril_mask = paddle.broadcast_to(tril_mask, mat.shape)
    vec = paddle.masked_select(mat, tril_mask).reshape(out_shape)
    return vec


class LKJCholesky(distribution.Distribution):
    """
    The LKJCholesky class represents the LKJ distribution over Cholesky factors of correlation matrices.
    This class implements the LKJ distribution over Cholesky factors of correlation matrices, as described in
    Lewandowski, Kurowicka, and Joe (2009). It supports two sampling methods: "onion" and "cvine".

    Args:
        dim (int): The dimension of the correlation matrices.
        concentration (float, optional): The concentration parameter of the LKJ distribution. Default is 1.0.
        sample_method (str, optional): The sampling method to use, either "onion" or "cvine". Default is "onion".

    Example:
        .. code-block:: python

            >>> import paddle

            >>> dim = 3
            >>> lkj = paddle.distribution.LKJCholesky(dim=dim)
            >>> sample = lkj.sample()
            >>> sample.shape
            [3, 3]
    """

    concentration: Tensor
    dtype: _DTypeLiteral
    dim: int
    sample_method: Literal["onion", "cvine"]

    def __init__(
        self,
        dim: int = 2,
        concentration: float = 1.0,
        sample_method: Literal["onion", "cvine"] = "onion",
    ) -> None:
        if not in_dynamic_mode():
            check_type(
                dim,
                "dim",
                (int, Variable, paddle.pir.Value),
                "LKJCholesky",
            )
            check_type(
                concentration,
                "concentration",
                (float, list, tuple, Variable, paddle.pir.Value),
                "LKJCholesky",
            )

        # Get/convert concentration/rate to tensor.
        if self._validate_args(concentration):
            self.concentration = concentration
            self.dtype = convert_dtype(concentration.dtype)
        else:
            [self.concentration] = self._to_tensor(concentration)
            self.dtype = paddle.get_default_dtype()

        self.dim = dim
        if not self.dim >= 2:
            raise ValueError(
                f"Expected dim greater than or equal to 2. Found dim={dim}."
            )
        elif not isinstance(self.dim, int):
            raise TypeError(f"Expected dim to be an integer. Found dim={dim}.")

        if in_dynamic_mode():
            if not paddle.all(self.concentration > 0):
                raise ValueError("The arg of `concentration` must be positive.")

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

    def _onion(self, sample_shape: Sequence[int]) -> Tensor:
        """Generate a sample using the "onion" method.

        Args:
            sample_shape (tuple): The shape of the samples to be generated.

        Returns:
            w (Tensor): The Cholesky factor of the sampled correlation matrix.
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
        # TODO: check if static graph can use fill_
        # u_hypersphere[..., 0, :].fill_(0.0)
        # u_hypersphere[..., 0, :] = 0.0
        u_hypersphere_other = u_hypersphere[..., 1:, :]
        zero_shape = (*tuple(u_hypersphere.shape[:-2]), 1, self.dim)
        zero_row = paddle.zeros(shape=zero_shape, dtype=u_hypersphere.dtype)
        u_hypersphere = paddle.concat([zero_row, u_hypersphere_other], axis=-2)

        w = paddle.sqrt(y) * u_hypersphere

        # Fill diagonal elements; clamp for numerical stability
        eps = paddle.finfo(w.dtype).tiny
        diag_elems = paddle.clip(1 - paddle.sum(w**2, axis=-1), min=eps).sqrt()

        w += paddle.diag_embed(diag_elems)
        return w

    def _cvine(self, sample_shape: Sequence[int]) -> Tensor:
        """Generate a sample using the "cvine" method.

        Args:
            sample_shape (tuple): The shape of the samples to be generated.

        Returns:
            r (Tensor): The Cholesky factor of the sampled correlation matrix.
        """

        # Sample beta and calculate partial correlations
        beta_sample = self._beta.sample(sample_shape).unsqueeze(-1)
        partial_correlation = 2 * beta_sample - 1

        if self.dim == 2:
            partial_correlation = partial_correlation.unsqueeze(-2)

        # Construct the lower triangular matrix from the partial correlations
        last_dim = self.dim * (self.dim - 1) // 2
        flatten_shape = last_dim * reduce(operator.mul, sample_shape, 1)
        if len(self.concentration.shape) != 0:
            flatten_shape *= self.concentration.shape[-1]

        partial_correlation = partial_correlation.reshape((flatten_shape,))
        partial_correlation = vec_to_tril_matrix(
            partial_correlation,
            self.dim,
            last_dim,
            flatten_shape,
            sample_shape,
            -1,
        )

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
        r = r * z1m_cumprod_sqrt_shifted
        return r

    def sample(self, sample_shape: Sequence[int] = []) -> Tensor:
        """Generate a sample using the specified sampling method."""
        if not isinstance(sample_shape, Sequence):
            raise TypeError('sample shape must be Sequence object.')

        if self.sample_method == "onion":
            res = self._onion(sample_shape)
        else:
            res = self._cvine(sample_shape)

        output_shape = list(sample_shape)
        output_shape.extend(self.concentration.shape)
        output_shape.extend([self.dim, self.dim])

        return res.reshape(output_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Compute the log probability density of the given Cholesky factor under the LKJ distribution.

        Args:
            value (Tensor): The Cholesky factor of the correlation matrix for which the log probability density is to be computed.

        Returns:
            log_prob (Tensor): The log probability density of the given Cholesky factor under the LKJ distribution.
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
