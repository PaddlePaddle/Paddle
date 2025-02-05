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


class Range(Constraint):
    def __init__(self, lower: float | Tensor, upper: float | Tensor) -> None:
        self._lower = lower
        self._upper = upper
        super().__init__()

    def __call__(self, value: Tensor) -> Tensor:
        return self._lower <= value <= self._upper


class Positive(Constraint):
    def __call__(self, value: Tensor) -> Tensor:
        return value >= 0.0


class Simplex(Constraint):
    def __call__(self, value: Tensor) -> Tensor:
        return paddle.all(value >= 0, axis=-1) and (
            (value.sum(-1) - 1).abs() < 1e-6
        )


real = Real()
positive = Positive()
simplex = Simplex()
