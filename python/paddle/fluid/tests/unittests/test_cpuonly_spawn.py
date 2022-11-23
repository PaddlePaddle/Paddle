# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist


class LinearNet(nn.Layer):

    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))


def train(print_result=False):
    # 1. initialize parallel environment
    dist.init_parallel_env()

    # 2. create data parallel layer & optimizer
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)

    loss_fn = nn.MSELoss()
    adam = opt.Adam(learning_rate=0.001, parameters=dp_layer.parameters())

    # 3. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    if print_result is True:
        print("loss:", loss.numpy())

    loss.backward()
    print("Grad is", layer._linear1.weight.grad)
    adam.step()
    adam.clear_grad()


class TestSpawn(unittest.TestCase):

    def test_spawn(self):
        dist.spawn(train, backend='gloo', nprocs=4)

    def test_wrong_backend(self):
        try:
            dist.spawn(train, backend='something', nprocs=4)
        except ValueError as e:
            self.assertEqual(type(e), ValueError)


if __name__ == '__main__':
    unittest.main()
