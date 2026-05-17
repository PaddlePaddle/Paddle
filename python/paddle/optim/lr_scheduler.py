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

from paddle.optimizer.lr import (
    CosineAnnealingDecay as CosineAnnealingLR,  # noqa: F401
    CosineAnnealingWarmRestarts as CosineAnnealingWarmRestarts,
    CyclicLR as CyclicLR,
    ExponentialDecay as ExponentialLR,  # noqa: F401
    LambdaDecay as LambdaLR,  # noqa: F401
    LinearLR as LinearLR,
    LRScheduler as LRScheduler,
    MultiplicativeDecay as MultiplicativeLR,  # noqa: F401
    MultiStepDecay as MultiStepLR,  # noqa: F401
    OneCycleLR as OneCycleLR,
    PiecewiseDecay as ConstantLR,  # noqa: F401
    ReduceOnPlateau as ReduceLROnPlateau,  # noqa: F401
    StepDecay as StepLR,  # noqa: F401
)
