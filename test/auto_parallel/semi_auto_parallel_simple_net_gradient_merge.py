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


class TestSimpleNetWithGradientMergeForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.set_device(self._backend)
        self.init_single_card_net_result()

    def run_dynamic_gradient_merge(self, layer, shard_input=False):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        # create loss
        loss_fn = nn.MSELoss()
        opt = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=layer.parameters()
        )
        # run forward and backward
        image, label = self.init_input_data()
        if shard_input:
            image = dist.shard_tensor(image, self._mesh, [dist.Shard(0)])

        for i in range(2):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()

        return loss, layer.parameters()

    def init_single_card_net_result(self):
        self.set_random_seed(self._seed)
        (
            self.base_loss,
            self.base_parameters,
        ) = self.run_dynamic_gradient_merge(DemoNet("gradient_merge_demo"))

    def test_dp_demo_net(self):
        self.set_random_seed(self._seed)
        (
            self.dp_loss,
            self.dp_parameters,
        ) = self.run_dynamic_gradient_merge(
            DemoNet("gradient_merge_dp_demo"),
            shard_input=True,
        )
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        for param, param_base in zip(self.dp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base, rtol=2e-4)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_mp_demo_net(self):
        self.set_random_seed(self._seed)
        mp_layer = dist.shard_layer(
            DemoNet("gradient_merge_mp_demo"), self._mesh, self.shard_fn
        )
        (
            self.mp_loss,
            self.mp_parameters,
        ) = self.run_dynamic_gradient_merge(mp_layer)

        self.check_tensor_eq(self.mp_loss, self.base_loss)
        for param, param_base in zip(self.mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def run_test_case(self):
        self.test_dp_demo_net()
        self.test_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetWithGradientMergeForSemiAutoParallel().run_test_case()
