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
    MPDemoNet,
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist
from paddle import nn

hook_triggered = False


def backward_hook():
    def trigger_hook():
        global hook_triggered
        hook_triggered = True

    return trigger_hook()


class TestSimpleNetWithGradientHookForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.set_device(self._backend)
        self.init_input_data()
        self.init_single_card_net_result()

    def run_dynamic(self, layer):
        loss_fn = nn.MSELoss()
        image = paddle.to_tensor(self.image)

        out = layer(image)
        label = paddle.to_tensor(self.label)
        loss = loss_fn(out, label)
        loss.backward()

    def test_gradient_hook(self):
        model = MPDemoNet(self.w0, self.w1, self._mesh)
        model.w0._register_grad_hook(backward_hook())
        self.run_dynamic(model)
        global hook_triggered
        assert hook_triggered

    def run_test_case(self):
        self.test_gradient_hook()


if __name__ == '__main__':
    TestSimpleNetWithGradientHookForSemiAutoParallel().run_test_case()
