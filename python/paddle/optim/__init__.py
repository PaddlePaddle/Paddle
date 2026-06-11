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
    from paddle.optimizer.adam import _AdamParameterConfig
    from paddle.optimizer.optimizer import _ParameterConfig

import sys as _sys
import warnings

from paddle.optimizer import (
    ASGD as ASGD,
    LBFGS as LBFGS,
    SGD as PaddleSGD,
    Adadelta as Adadelta,
    Adagrad as PaddleAdagrad,
    Adam as Adam,
    Adamax as Adamax,
    AdamW as PaddleAdamW,
    Muon as Muon,
    NAdam as NAdam,
    Optimizer as Optimizer,
    RAdam as RAdam,
    RMSProp as RMSProp,
    Rprop as Rprop,
    adadelta,
    adagrad,
    adam,
    adamax,
    adamw,
    asgd,
    lbfgs,
    muon,
    nadam,
    optimizer,
    radam,
    rmsprop,
    rprop,
    sgd,
)

from . import lr_scheduler  # noqa: F401


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
    ) -> None:
        warnings.warn(
            "lr_decay, foreach are currently not supported in Adagrad and will be ignored. "
            "The parameters are reserved for future implementation."
        )
        super().__init__(
            learning_rate=lr,
            epsilon=eps,
            parameters=params,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
        )


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


class SGD(PaddleSGD):
    def __init__(
        self,
        params: Sequence[Tensor] | Sequence[_ParameterConfig] | None = None,
        lr: float | Tensor = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float | Tensor = 0,
        nesterov: bool = False,
    ) -> None:
        warnings.warn(
            "momentum, dampening, nesterov are currently not supported in SGD and will be ignored. "
            "The parameters are reserved for future implementation."
        )
        super().__init__(
            learning_rate=lr,
            parameters=params,
            weight_decay=weight_decay,
        )


_sys.modules['paddle.optim.adadelta'] = adadelta
_sys.modules['paddle.optim.adagrad'] = adagrad
_sys.modules['paddle.optim.adam'] = adam
_sys.modules['paddle.optim.adamax'] = adamax
_sys.modules['paddle.optim.adamw'] = adamw
_sys.modules['paddle.optim.asgd'] = asgd
_sys.modules['paddle.optim.lbfgs'] = lbfgs
_sys.modules['paddle.optim.muon'] = muon
_sys.modules['paddle.optim.nadam'] = nadam
_sys.modules['paddle.optim.optimizer'] = optimizer
_sys.modules['paddle.optim.radam'] = radam
_sys.modules['paddle.optim.rmsprop'] = rmsprop
_sys.modules['paddle.optim.rprop'] = rprop
_sys.modules['paddle.optim.sgd'] = sgd

__all__ = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "Adamax",
    "AdamW",
    "ASGD",
    "LBFGS",
    "Muon",
    "NAdam",
    "Optimizer",
    "RAdam",
    "RMSProp",
    "Rprop",
    "SGD",
]
