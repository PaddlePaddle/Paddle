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

import random

import numpy as np

import paddle

paddle.set_device('cpu')
# paddle.set_device('gpu:1')

paddle.seed(1010)
np.random.seed(1010)
random.seed(1010)


class TinyModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        return self.linear(x)


def loss_fn(x):
    return x.mean()


def train_step(net, x, loss_fn, opt):
    out = net(x)
    loss = loss_fn(out)
    loss.backward()
    opt.step()
    opt.clear_grad()
    return loss


x = paddle.randn([100, 10])
net = TinyModel()
sgd = paddle.optimizer.SGD(0.001, parameters=net.parameters())

train_step = paddle.jit.to_static(train_step)

train_step(net, x, loss_fn, sgd)
