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

__all__ = [
    'Adadelta', 'AdadeltaOptimizer', 'Adagrad', 'AdagradOptimizer', 'Adam',
    'Adamax', 'AdamW', 'DecayedAdagrad', 'DecayedAdagradOptimizer', 'Dpsgd',
    'DpsgdOptimizer', 'ExponentialMovingAverage', 'Ftrl', 'FtrlOptimizer',
    'LookaheadOptimizer', 'ModelAverage', 'Momentum', 'MomentumOptimizer',
    'RMSProp', 'SGD', 'SGDOptimizer', 'Optimizer', '_LRScheduler', 'NoamLR',
    'PiecewiseLR', 'NaturalExpLR', 'InverseTimeLR', 'PolynomialLR',
    'LinearLrWarmup', 'ExponentialLR', 'MultiStepLR', 'StepLR', 'LambdaLR',
    'ReduceLROnPlateau', 'CosineAnnealingLR'
]


from ..fluid.optimizer import Momentum, Adagrad, Dpsgd, DecayedAdagrad, Ftrl,\
            AdagradOptimizer, DpsgdOptimizer, DecayedAdagradOptimizer, \
            FtrlOptimizer, AdadeltaOptimizer, ModelAverage, \
            ExponentialMovingAverage, LookaheadOptimizer

from .optimizer import Optimizer
from .adam import Adam
from .adamw import AdamW
from .adamax import Adamax
from .rmsprop import RMSProp
from .adadelta import Adadelta
from .sgd import SGD
from .momentum import Momentum

from . import lr_scheduler
from .lr_scheduler import _LRScheduler, NoamLR, PiecewiseLR, NaturalExpLR, InverseTimeLR, PolynomialLR, \
            LinearLrWarmup, ExponentialLR, MultiStepLR, StepLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingLR
