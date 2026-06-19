# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle.optimizer.adagrad import _AdagradParameterConfig

import warnings

from paddle.optimizer import Adagrad as PaddleAdagrad


class Adagrad(PaddleAdagrad):
    def __init__(
        self,
        params: Sequence[Tensor] | Sequence[_AdagradParameterConfig] | None,
        lr: float | Tensor = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        foreach: bool | None = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        if (
            lr_decay != 0
            or foreach is not None
            or differentiable is True
            or fused is not None
        ):
            warnings.warn(
                "lr_decay, foreach, differentiable, fused are currently not supported in Adagrad and will be ignored. "
                "The parameters are reserved for future implementation."
            )
        super().__init__(
            learning_rate=lr,
            epsilon=eps,
            parameters=params,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            maximize=maximize,
        )
