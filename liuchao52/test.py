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
import paddle.nn as nn
import paddle.optimizer as opt
from paddle.static import InputSpec

IMAGE_SIZE = 16
CLASS_NUM = 10

# paddle.framework.set_flags({'FLAGS_enable_new_ir_in_executor': True})
paddle.framework.set_flags({'FLAGS_use_cinn': False})
# paddle.framework.set_flags(
#     {'FLAGS_cinn_subgraph_graphviz_dir': '/home/workspace/dev3/Paddle/build/cinn_graph/'})


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        # net, input_spec=[InputSpec(shape=[None, 1], dtype='float32'), InputSpec(shape=[None, 1], dtype='float32')], build_strategy=build_strategy, full_graph=True
        net,
        input_spec=[
            InputSpec(shape=[16, 1], dtype='float32'),
            InputSpec(shape=[16, 1], dtype='float32'),
        ],
        build_strategy=build_strategy,
        full_graph=True,
    )


class TestNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    # @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 1], dtype='float32'), InputSpec(shape=[None, 1], dtype='float32')], full_graph=True)
    def forward(self, x, y):
        return x - y
        # return x - paddle.exp(y)


def train(layer, loss_fn, opt):
    for batch_id in range(0, 3):
        input_x = paddle.randn([(IMAGE_SIZE + batch_id), 1], dtype='float32')
        input_y = paddle.randn([(IMAGE_SIZE + batch_id), 1], dtype='float32')
        label = paddle.randint(0, CLASS_NUM, (IMAGE_SIZE + batch_id, 1)).astype(
            'int64'
        )
        out = layer(input_x, input_y)
        print("batch {}: out = {}".format(batch_id, np.mean(out.numpy())))


# create network
layer = TestNet()
layer = apply_to_static(layer, True)
layer.eval()
loss_fn = nn.CrossEntropyLoss()
adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

# train
train(layer, loss_fn, adam)
