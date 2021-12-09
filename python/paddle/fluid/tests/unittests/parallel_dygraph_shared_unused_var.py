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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.dygraph.base import to_variable
from test_dist_base import runtime_main, TestParallelDyGraphRunnerBase

np.random.seed(2021)
paddle.seed(1024)


class SimpleNet(fluid.Layer):
    def __init__(self):
        # bias is unused parameters, and it share with net_a
        super(SimpleNet, self).__init__()
        self.net_a = Linear(input_dim=10, output_dim=5)
        self.net_b = Linear(10, 10)
        self.bias = self.net_a.bias

    def forward(self, x):
        return self.net_b(x)


batch_size = 4
batch_num = 1000


def fake_sample_reader():
    def __reader__():
        for i in range(batch_num):
            x_data = np.random.random_sample((10, )).astype('float32')
            yield x_data

    return __reader__


class TestSimpleNet(TestParallelDyGraphRunnerBase):
    def get_model(self):
        model = SimpleNet()
        train_reader = paddle.batch(
            fake_sample_reader(), batch_size=batch_size, drop_last=True)
        optimizer = paddle.optimizer.SGD(learning_rate=0.001,
                                         parameters=model.parameters())
        return model, train_reader, optimizer

    def run_one_loop(self, model, optimizer, batch):
        x_data = np.array([x for x in batch])
        x_data = x_data.reshape((-1, 10))
        x = to_variable(x_data)
        out = model(x)
        loss = out.sum() / len(batch)
        return loss


if __name__ == "__main__":
    runtime_main(TestSimpleNet)
