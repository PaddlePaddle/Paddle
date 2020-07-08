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
    'Adamax', 'AdamaxOptimizer', 'AdamOptimizer', 'DecayedAdagrad',
    'DecayedAdagradOptimizer', 'DGCMomentumOptimizer', 'Dpsgd',
    'DpsgdOptimizer', 'ExponentialMovingAverage', 'Ftrl', 'FtrlOptimizer',
    'LambOptimizer', 'LarsMomentum', 'LarsMomentumOptimizer',
    'LookaheadOptimizer', 'ModelAverage', 'Momentum', 'MomentumOptimizer',
    'PipelineOptimizer', 'RecomputeOptimizer', 'RMSPropOptimizer', 'SGD',
    'SGDOptimizer'
]


from ..fluid.optimizer import  SGD, Momentum, Adagrad, Adam, Adamax, Dpsgd, DecayedAdagrad, \
            Ftrl, SGDOptimizer, MomentumOptimizer, AdagradOptimizer, \
            AdamOptimizer, AdamaxOptimizer, DpsgdOptimizer, \
            DecayedAdagradOptimizer, RMSPropOptimizer, FtrlOptimizer, Adadelta, \
            AdadeltaOptimizer, ModelAverage, LarsMomentum, \
            LarsMomentumOptimizer, DGCMomentumOptimizer, LambOptimizer, \
            ExponentialMovingAverage, PipelineOptimizer, LookaheadOptimizer, \
            RecomputeOptimizer
