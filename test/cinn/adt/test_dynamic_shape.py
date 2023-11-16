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
from paddle import nn
from paddle.static import InputSpec

IMAGE_SIZE = 16
CLASS_NUM = 10


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=[
            InputSpec(shape=[None, 1], dtype='float32'),
            InputSpec(shape=[None, 1], dtype='float32'),
        ],
        build_strategy=build_strategy,
        full_graph=True,
    )


class TestNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x - paddle.exp(y)


def train(layer):
    for batch_id in range(0, 3):
        input_x = paddle.randn([(IMAGE_SIZE + batch_id), 1], dtype='float32')
        input_y = paddle.randn([(IMAGE_SIZE + batch_id), 1], dtype='float32')
        out = layer(input_x, input_y)
        print(f"batch {batch_id}: out = {np.mean(out.numpy())}")


# create network
layer = TestNet()
layer = apply_to_static(layer, True)
layer.eval()
train(layer)
