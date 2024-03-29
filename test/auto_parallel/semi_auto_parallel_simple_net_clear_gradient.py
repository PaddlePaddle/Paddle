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

import os

from semi_auto_parallel_simple_net import (
    DemoNet,
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist
from paddle import nn


class TestSimpleNetWithClearGradientForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.set_device(self._backend)

    def run_dynamic_clear_gradient(self, layer, shard_input=False):
        # create loss
        loss_fn = nn.MSELoss()
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        # run forward and backward
        for _ in range(5):
            image, label = self.init_input_data()
            if shard_input:
                image = dist.shard_tensor(image, self._mesh, [dist.Shard(0)])
            out = layer(image)
            loss = loss_fn(out, label)

            loss.backward()
            opt.step()
            opt.clear_grad()
            for param in layer.parameters():
                param.clear_gradient()
                param.clear_gradient(False)

    def test_demo_net(self):
        mp_layer = dist.shard_layer(
            DemoNet("clear_gradient_demo"),
            self._mesh,
            self.shard_fn,
        )
        self.run_dynamic_clear_gradient(mp_layer)

    def run_test_case(self):
        self.test_demo_net()


if __name__ == '__main__':
    TestSimpleNetWithClearGradientForSemiAutoParallel().run_test_case()
