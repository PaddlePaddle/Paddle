# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from paddle.trainer_config_helpers.activations import *

__all__ = [
    "Base", "Tanh", "Sigmoid", "Softmax", "Identity", "Linear",
    'SequenceSoftmax', "Exp", "Relu", "BRelu", "SoftRelu", "STanh", "Abs",
    "Square", "Log"
]

Base = BaseActivation
Tanh = TanhActivation
Sigmoid = SigmoidActivation
Softmax = SoftmaxActivation
SequenceSoftmax = SequenceSoftmaxActivation
Identity = IdentityActivation
Linear = Identity
Relu = ReluActivation
BRelu = BReluActivation
SoftRelu = SoftReluActivation
STanh = STanhActivation
Abs = AbsActivation
Square = SquareActivation
Exp = ExpActivation
Log = LogActivation
