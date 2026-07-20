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

import warnings
from typing import TYPE_CHECKING

from paddle.optimizer import SGD as PaddleSGD

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle.optimizer.optimizer import _ParameterConfig


class SGD(PaddleSGD):
    def __init__(
        self,
        params: Sequence[Tensor] | Sequence[_ParameterConfig] | None = None,
        lr: float | Tensor = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float | Tensor = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        if (
            momentum != 0
            or dampening != 0
            or nesterov is True
            or foreach is not None
            or differentiable is True
            or fused is not None
        ):
            warnings.warn(
                "momentum, dampening, nesterov, foreach, differentiable, fused are currently not supported in SGD and will be ignored. "
                "The parameters are reserved for future implementation."
            )
        super().__init__(
            learning_rate=lr,
            parameters=params,
            weight_decay=weight_decay,
            maximize=maximize,
        )
