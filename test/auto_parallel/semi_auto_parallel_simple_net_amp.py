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
    DPDemoNet,
    MPDemoNet,
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist
from paddle import nn


class TestSimpleNetWithAmpForSemiAutoParallel(TestSimpleNetForSemiAutoParallel):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.set_device(self._backend)
        self.init_input_data()
        self.init_single_card_net_result()

    def run_dynamic_amp(self, layer, level='O1'):
        if level == 'O2':
            layer = paddle.amp.decorate(models=layer, level='O2')
        # create loss
        loss_fn = nn.MSELoss()
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        # run forward and backward
        image = paddle.to_tensor(self.image)

        with paddle.amp.auto_cast(level=level):
            out = layer(image)
            label = paddle.to_tensor(self.label)
            loss = loss_fn(out, label)

        scaled = scaler.scale(loss)
        scaled.backward()
        return loss, layer.w0.grad, layer.w1.grad

    def init_single_card_net_result(self):
        (
            self.base_loss,
            self.base_w0_grad,
            self.base_w1_grad,
        ) = self.run_dynamic_amp(DemoNet(self.w0, self.w1))

    def test_dp_demo_net(self, level):
        self.dp_loss, self.dp_w0_grad, self.dp_w1_grad = self.run_dynamic_amp(
            DPDemoNet(self.w0, self.w1, self._mesh), level
        )
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        self.check_tensor_eq(self.dp_w0_grad, self.base_w0_grad)
        self.check_tensor_eq(self.dp_w1_grad, self.base_w1_grad)

    def test_mp_demo_net(self, level):
        self.mp_loss, self.mp_w0_grad, self.mp_w1_grad = self.run_dynamic_amp(
            MPDemoNet(self.w0, self.w1, self._mesh), level
        )
        self.check_tensor_eq(self.mp_loss, self.base_loss)
        self.check_tensor_eq(self.mp_w0_grad, self.base_w0_grad)
        self.check_tensor_eq(self.mp_w1_grad, self.base_w1_grad)

    def run_test_case(self):
        self.test_dp_demo_net('O1')
        self.test_mp_demo_net('O1')
        # self.test_dp_demo_net('O2')
        # self.test_mp_demo_net('O2')


if __name__ == '__main__':
    TestSimpleNetWithAmpForSemiAutoParallel().run_test_case()
