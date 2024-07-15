#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from . import lr  # noqa: F401
from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .adamax import Adamax
from .adamw import AdamW
from .asgd import ASGD
from .lamb import Lamb
from .lbfgs import LBFGS
from .momentum import Momentum
from .nadam import NAdam
from .optimizer import Optimizer
from .radam import RAdam
from .rmsprop import RMSProp
from .rprop import Rprop
from .sgd import SGD

__all__ = [
    'Optimizer',
    'Adagrad',
    'Adam',
    'AdamW',
    'Adamax',
    'ASGD',
    'RAdam',
    'RMSProp',
    'Adadelta',
    'SGD',
    'Rprop',
    'Momentum',
    'NAdam',
    'Lamb',
    'LBFGS',
]
