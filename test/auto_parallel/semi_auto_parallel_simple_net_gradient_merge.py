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


class TestSimpleNetWithGradientMergeForSemiAutoParallel(
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

    def run_dynamic_gradient_merge(self, layer):
        # create loss
        loss_fn = nn.MSELoss()
        # run forward and backward
        image = paddle.to_tensor(self.image)

        for i in range(2):
            out = layer(image)
            label = paddle.to_tensor(self.label)
            loss = loss_fn(out, label)
            loss.backward()

        return loss, layer.w0.grad, layer.w1.grad

    def init_single_card_net_result(self):
        (
            self.base_loss,
            self.base_w0_grad,
            self.base_w1_grad,
        ) = self.run_dynamic_gradient_merge(DemoNet(self.w0, self.w1))

    def test_dp_demo_net(self):
        (
            self.dp_loss,
            self.dp_w0_grad,
            self.dp_w1_grad,
        ) = self.run_dynamic_gradient_merge(
            DPDemoNet(self.w0, self.w1, self._mesh)
        )
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        self.check_tensor_eq(self.dp_w0_grad, self.base_w0_grad)
        self.check_tensor_eq(self.dp_w1_grad, self.base_w1_grad)

    def test_mp_demo_net(self):
        (
            self.mp_loss,
            self.mp_w0_grad,
            self.mp_w1_grad,
        ) = self.run_dynamic_gradient_merge(
            MPDemoNet(self.w0, self.w1, self._mesh)
        )
        self.check_tensor_eq(self.mp_loss, self.base_loss)
        self.check_tensor_eq(self.mp_w0_grad, self.base_w0_grad)
        self.check_tensor_eq(self.mp_w1_grad, self.base_w1_grad)

    def run_test_case(self):
        self.test_dp_demo_net()
        self.test_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetWithGradientMergeForSemiAutoParallel().run_test_case()
