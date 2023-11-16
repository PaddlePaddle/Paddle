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


class TestSimpleNetWithAmpForSemiAutoParallel(TestSimpleNetForSemiAutoParallel):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._use_master_grad = bool(eval(os.getenv("use_master_grad")))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.set_device(self._backend)
        self.init_single_card_net_result()

    def check_tensor_eq(self, tensor_a, tensor_b):
        super().check_tensor_eq(tensor_a, tensor_b, rtol=1e-5, atol=1e-7)

    def run_dynamic_amp(self, layer, level='O1', shard_input=False):
        # create loss
        loss_fn = nn.MSELoss()
        opt = paddle.optimizer.AdamW(
            learning_rate=0.1, parameters=layer.parameters()
        )

        if level == 'O2':
            layer, opt = paddle.amp.decorate(
                models=layer,
                level='O2',
                master_grad=self._use_master_grad,
                optimizers=opt,
                dtype=self._dtype,
            )

        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        # run forward and backward
        for _ in range(5):
            image, label = self.init_input_data()
            if shard_input:
                image = dist.shard_tensor(
                    image,
                    dist_attr=dist.DistAttr(
                        mesh=self._mesh, sharding_specs=['x', None]
                    ),
                )

            with paddle.amp.auto_cast(level=level):
                out = layer(image)
                loss = loss_fn(out, label)

            scaled = scaler.scale(loss)
            scaled.backward()
            opt.step()
            opt.clear_grad()

        return loss, layer.parameters()

    def init_single_card_net_result(self):
        self.set_random_seed(self._seed)

        (
            self.base_loss_o1,
            self.base_parameters_o1,
        ) = self.run_dynamic_amp(DemoNet('demo_weight_O1'), 'O1')

        self.set_random_seed(self._seed)
        (
            self.base_loss_o2,
            self.base_parameters_o2,
        ) = self.run_dynamic_amp(DemoNet('demo_weight_O2'), 'O2')

    def test_dp_demo_net(self):
        self.set_random_seed(self._seed)
        (
            self.dp_loss_o1,
            self.dp_parameters_o1,
        ) = self.run_dynamic_amp(
            DemoNet('dp_demo_weight_O1'), 'O1', shard_input=True
        )
        self.check_tensor_eq(self.dp_loss_o1, self.base_loss_o1)
        for param, param_base in zip(
            self.dp_parameters_o1, self.base_parameters_o1
        ):
            # self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

        self.set_random_seed(self._seed)
        (
            self.dp_loss_o2,
            self.dp_parameters_o2,
        ) = self.run_dynamic_amp(DemoNet('dp_demo_weight_O2'), 'O2')
        self.check_tensor_eq(self.dp_loss_o2, self.base_loss_o2)
        for param, param_base in zip(
            self.dp_parameters_o2, self.base_parameters_o2
        ):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_mp_demo_net(self):
        self.set_random_seed(self._seed)
        mp_layer_o1 = dist.shard_layer(
            DemoNet("mp_demo_weight_O1"), self._mesh, self.shard_fn
        )
        (
            self.mp_loss_o1,
            self.mp_parameters_o1,
        ) = self.run_dynamic_amp(mp_layer_o1, 'O1')
        self.check_tensor_eq(self.mp_loss_o1, self.base_loss_o1)
        for param, param_base in zip(
            self.mp_parameters_o1, self.base_parameters_o1
        ):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

        self.set_random_seed(self._seed)
        mp_layer_o2 = dist.shard_layer(
            DemoNet("mp_demo_weight_O2"), self._mesh, self.shard_fn
        )
        (
            self.mp_loss_o2,
            self.mp_parameters_o2,
        ) = self.run_dynamic_amp(mp_layer_o2, 'O2')
        self.check_tensor_eq(self.mp_loss_o2, self.base_loss_o2)
        for param, param_base in zip(
            self.mp_parameters_o2, self.base_parameters_o2
        ):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def run_test_case(self):
        self.test_dp_demo_net()
        self.test_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetWithAmpForSemiAutoParallel().run_test_case()
