# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle.distributed.fleet.layers.mpu import mp_ops

label = np.random.randint(0, 1000, size=(1024, 1), dtype='int32')
ignore_index = label[0][0]

local_elements = 1000
input0 = np.random.uniform(
    low=-40.0, high=40.0, size=(1024, local_elements)
).astype("float32")

loss, softmax = mp_ops._c_softmax_with_cross_entropy(
    paddle.to_tensor(input0),
    paddle.to_tensor(label),
    ignore_index=ignore_index,
    return_softmax=True,
)

print(softmax)
