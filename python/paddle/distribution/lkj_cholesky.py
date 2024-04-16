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


class LKJCholesky(distribution.Distribution):
    def __init__(self, dim, concentration=1.0):
        # need dim > 2, TODO add assert
        self.dim = dim

        self.concentration = paddle.to_tensor(concentration)
        batch_shape = self.concentration.shape
        event_shape = paddle.to_tensor((dim, dim))

        # This is used to draw vectorized samples from the beta distribution in Sec. 3.2 of [1].
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        offset = paddle.arange(
            self.dim - 1,
            dtype=self.concentration.dtype,
        )
        offset = paddle.concat([paddle.zeros((1,), dtype=offset.dtype), offset])
        beta_conc1 = offset + 0.5
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
        self._beta = Beta(beta_conc1, beta_conc0)

        super().__init__(batch_shape, event_shape)

    def sample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = paddle.to_tensor([])
        y = self._beta.sample(sample_shape).unsqueeze(-1)
        u_normal = paddle.randn(
            self._extend_shape(sample_shape), dtype=y.dtype
        ).tril(-1)
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

    def log_prob(self, value):
        diag_elems = paddle.diagonal(value, offset=0, axis1=-1, axis2=-2)[
            ..., 1:
        ]
        order = paddle.arange(2, self.dim + 1, dtype=self.concentration.dtype)
        order = 2 * (self.concentration - 1).unsqueeze(-1) + self.dim - order

        unnormalized_log_pdf = paddle.sum(
            order * paddle.log(diag_elems), axis=-1
        )
        # Compute normalization constant (page 1999 of [1])
        dm1 = self.dim - 1

        alpha = self.concentration + 0.5 * dm1
        denominator = paddle.lgamma(alpha) * dm1
        numerator = self.mvlgamma(alpha - 0.5, dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * dm1 * math.log(math.pi)

        normalize_term = pi_constant + numerator - denominator
        return unnormalized_log_pdf - normalize_term

    def mvlgamma(self, a, p):
        """
        :param a: A scalar or tensor of shape (...,)
        :param p: An integer representing the dimension of the multivariate gamma function
        :return: The result of the multivariate gamma function for each element in a
        """
        p_float = float(p)
        order = paddle.arange(0, p_float, dtype=a.dtype)
        return paddle.sum(paddle.lgamma(a.unsqueeze(-1) - 0.5 * order), axis=-1)
