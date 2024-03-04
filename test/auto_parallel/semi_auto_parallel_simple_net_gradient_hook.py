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

import numpy as np
from semi_auto_parallel_simple_net import (
    DemoNet,
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist
from paddle import nn

hook_triggered = False


def backward_hook():
    def trigger_hook(grad):
        global hook_triggered
        hook_triggered = True
        assert grad.is_dist()
        return paddle.scale(grad, 1.0)

    return trigger_hook


class TestSimpleNetWithGradientHookForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.set_device(self._backend)

    def run_dynamic(self, layer):
        image, label = self.init_input_data()
        loss_fn = nn.MSELoss()

        out = layer(image)
        loss = loss_fn(out, label)
        loss.backward()

    def test_register_grad_hook(self):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        model = dist.shard_layer(
            DemoNet("mp_demo_register_grad_hook"), self._mesh, self.shard_fn
        )
        model.parameters()[0]._register_grad_hook(backward_hook())

        self.run_dynamic(model)
        global hook_triggered
        assert hook_triggered
        hook_triggered = False

    def test_register_hook(self):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        model = dist.shard_layer(
            DemoNet("mp_demo_register_hook"), self._mesh, self.shard_fn
        )
        model.parameters()[0].register_hook(backward_hook())

        self.run_dynamic(model)
        global hook_triggered
        assert hook_triggered
        hook_triggered = False

    def run_test_case(self):
        self.test_register_grad_hook()
        self.test_register_hook()


if __name__ == '__main__':
    TestSimpleNetWithGradientHookForSemiAutoParallel().run_test_case()
