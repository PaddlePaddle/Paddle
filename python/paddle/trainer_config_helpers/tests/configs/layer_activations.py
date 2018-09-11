#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
'''
Test all activations.
'''

from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

din = data_layer(name='input', size=100)

acts = [
    TanhActivation, SigmoidActivation, SoftmaxActivation, IdentityActivation,
    LinearActivation, ExpActivation, ReluActivation, BReluActivation,
    SoftReluActivation, STanhActivation, AbsActivation, SquareActivation
]

outputs([
    fc_layer(
        input=din, size=100, act=act(), name="layer_%d" % i)
    for i, act in enumerate(acts)
])
