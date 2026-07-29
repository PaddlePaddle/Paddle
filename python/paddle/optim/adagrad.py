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

from paddle.optimizer import Adagrad as PaddleAdagrad

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from paddle import Tensor
    from paddle.optimizer.adagrad import _AdagradParameterConfig


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
        if foreach is not None or differentiable is True or fused is not None:
            warnings.warn(
                "foreach, differentiable, fused are currently not supported in Adagrad and will be ignored. "
                "The parameters are reserved for future implementation."
            )
        self._lr_decay = None
        if lr_decay != 0.0:
            self._lr_decay = lr_decay
            self._step = -1
        super().__init__(
            learning_rate=lr,
            epsilon=eps,
            parameters=params,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            maximize=maximize,
        )

    def state_dict(self) -> dict[str, Tensor]:
        state_dict = super().state_dict()
        if self._lr_decay is not None:
            state_dict['step'] = self._step
        return state_dict

    def set_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        state_dict = state_dict.copy()
        if "step" in state_dict:
            if self._lr_decay is not None:
                self._step = state_dict["step"]
            state_dict.pop("step")
        return super().set_state_dict(state_dict)

    def _create_param_lr(self, param_and_grad):
        param_lr = super()._create_param_lr(param_and_grad)
        if self._lr_decay is not None:
            param_lr = param_lr / (1.0 + self._step * self._lr_decay)
        return param_lr

    def step(
        self, closure: Callable[[], Tensor] | None = None
    ) -> Tensor | None:
        if self._lr_decay is not None:
            self._step += 1
        return super().step(closure)
