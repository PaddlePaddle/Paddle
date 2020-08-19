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
    'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'DecayedAdagrad', 'AdamW'
    'DGCMomentumOptimizer', 'Dpsgd', 'DpsgdOptimizer',
    'ExponentialMovingAverage', 'Ftrl', 'Lamb', 'LarsMomentum',
    'LookaheadOptimizer', 'ModelAverage', 'Momentum', 'PipelineOptimizer',
    'RecomputeOptimizer', 'RMSProp', 'SGD', 'Optimizer'
]


from ..fluid.optimizer import  SGD, Momentum, Adagrad, Dpsgd, DecayedAdagrad, \
            Ftrl, Adadelta, Lamb, RMSProp, \
            ModelAverage, LarsMomentum, DGCMomentumOptimizer, \
            ExponentialMovingAverage, PipelineOptimizer, LookaheadOptimizer, \
            RecomputeOptimizer

from .optimizer import Optimizer
from .adam import Adam
from .adamw import AdamW
from .adamax import adamax
from .rmsprop import RMSProp
