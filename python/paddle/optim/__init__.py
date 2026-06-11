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

import sys as _sys

from paddle.optimizer import (
    ASGD as ASGD,
    LBFGS as LBFGS,
    SGD as SGD,
    Adadelta as Adadelta,
    Adagrad as Adagrad,
    Adam as Adam,
    Adamax as Adamax,
    AdamW as AdamW,
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
