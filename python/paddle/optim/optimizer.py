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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor
    from paddle.optimizer.optimizer import _ParameterConfig

from paddle.optimizer import Optimizer as PaddleOptimizer


class Optimizer(PaddleOptimizer):
    def __init__(
        self,
        params: Sequence[Tensor] | Sequence[_ParameterConfig] | None,
        defaults: dict[str, Any],
    ) -> None:
        lr = defaults.pop('lr', None)
        learning_rate = defaults.pop('learning_rate', None)
        if lr is not None and learning_rate is not None:
            raise ValueError(
                "Cannot specify both 'lr' and 'learning_rate' in defaults."
            )
        lr = lr if lr is not None else learning_rate

        weight_decay = defaults.pop('weight_decay', None)
        grad_clip = defaults.pop('grad_clip', None)
        maximize = defaults.pop('maximize', False)

        super().__init__(
            learning_rate=lr,
            parameters=params,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            maximize=maximize,
        )
