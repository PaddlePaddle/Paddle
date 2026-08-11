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
from __future__ import annotations

from typing import TYPE_CHECKING

import paddle

if TYPE_CHECKING:
    from paddle import Tensor


class Constraint:
    """Constraint condition for random variable."""

    def __call__(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def check(self, value: Tensor) -> Tensor:
        return self(value)


class Real(Constraint):
    def __call__(self, value: Tensor) -> Tensor:
        return value == value


class RealVector(Constraint):
    event_dim = 1

    def __call__(self, value: Tensor) -> Tensor:
        if value.dim() < 1:
            return paddle.zeros(value.shape[:-1], dtype='bool')
        return (value == value).reshape((*value.shape[:-1], -1)).all(-1)


class Range(Constraint):
    def __init__(self, lower: float | Tensor, upper: float | Tensor) -> None:
        self._lower = lower
        self._upper = upper
        super().__init__()

    def __call__(self, value: Tensor) -> Tensor:
        return self._lower <= value <= self._upper


class IntegerInterval(Constraint):
    event_dim = 0
    is_discrete = True

    def __init__(self, lower: int, upper: int) -> None:
        self._lower = lower
        self._upper = upper
        super().__init__()

    def __call__(self, value: Tensor) -> Tensor:
        return (
            (value >= self._lower) & (value <= self._upper) & (value % 1 == 0)
        )


class Positive(Constraint):
    def __call__(self, value: Tensor) -> Tensor:
        return value >= 0.0


class LowerTriangular(Constraint):
    event_dim = 2

    def __call__(self, value: Tensor) -> Tensor:
        if value.dim() < 2:
            return paddle.zeros(value.shape[:-2], dtype='bool')
        value_tril = paddle.tril(value)
        return (value_tril == value).reshape((*value.shape[:-2], -1)).all(-1)


class LowerCholesky(Constraint):
    event_dim = 2

    def __call__(self, value: Tensor) -> Tensor:
        if value.dim() < 2:
            return paddle.zeros(value.shape[:-2], dtype='bool')
        value_tril = paddle.tril(value)
        lower_triangular = (
            (value_tril == value).reshape((*value.shape[:-2], -1)).all(-1)
        )
        positive_diagonal = (value.diagonal(axis1=-2, axis2=-1) > 0).all(-1)
        return lower_triangular & positive_diagonal


class Square(Constraint):
    event_dim = 2

    def __call__(self, value: Tensor) -> Tensor:
        if value.dim() < 2:
            return paddle.full_like(value.sum(), False, dtype='bool')
        batch_value = value.reshape((*value.shape[:-2], -1)).sum(-1)
        return paddle.full_like(
            batch_value, value.shape[-2] == value.shape[-1], dtype='bool'
        )


class Symmetric(Square):
    def __call__(self, value: Tensor) -> Tensor:
        square_check = super().__call__(value)
        if value.dim() < 2:
            return square_check
        if value.shape[-2] != value.shape[-1]:
            return square_check
        return square_check & paddle.isclose(value, value.mT, atol=1e-6).all(
            -2
        ).all(-1)


class PositiveDefinite(Symmetric):
    def __call__(self, value: Tensor) -> Tensor:
        if value.dim() < 2:
            return paddle.zeros(value.shape[:-2], dtype='bool')
        sym_check = super().__call__(value)
        if value.shape[-2] != value.shape[-1]:
            return sym_check
        return sym_check & (paddle.linalg.eigvalsh(value) > 0).all(-1)


class Simplex(Constraint):
    def __call__(self, value: Tensor) -> Tensor:
        return paddle.all(value >= 0, axis=-1) and (
            (value.sum(-1) - 1).abs() < 1e-6
        )


real = Real()
real_vector = RealVector()
integer_interval = IntegerInterval
positive = Positive()
lower_triangular = LowerTriangular()
lower_cholesky = LowerCholesky()
square = Square()
symmetric = Symmetric()
positive_definite = PositiveDefinite()
simplex = Simplex()
