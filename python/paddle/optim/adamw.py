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
    from paddle.optimizer.adam import _AdamParameterConfig


from paddle.optimizer import AdamW as PaddleAdamW


class AdamW(PaddleAdamW):
    def __init__(
        self,
        params: Sequence[Tensor] | Sequence[_AdamParameterConfig] | None,
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(
            learning_rate=lr,
            beta1=betas[0],
            beta2=betas[1],
            epsilon=eps,
            parameters=params,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
