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


class TestSimpleNetWithAmpForSemiAutoParallel(TestSimpleNetForSemiAutoParallel):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._use_adam = eval(os.getenv("use_adam"))
        self._use_master_grad = bool(eval(os.getenv("use_master_grad")))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.set_device(self._backend)
        self.init_single_card_net_result()

    def check_tensor_eq(self, tensor_a, tensor_b):
        super().check_tensor_eq(tensor_a, tensor_b, rtol=1e-5, atol=1e-7)

    def run_dynamic_amp(
        self, layer, level='O1', shard_input=False, run_dist=False
    ):
        # create loss
        loss_fn = nn.MSELoss()
        if self._use_adam:
            opt = paddle.optimizer.Adam(
                learning_rate=0.001, parameters=layer.parameters()
            )
        else:
            opt = paddle.optimizer.AdamW(
                learning_rate=0.001, parameters=layer.parameters()
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
        if run_dist:
            scaler = dist.shard_scaler(scaler)
        # run forward and backward
        for _ in range(2):
            image, label = self.init_input_data()
            if shard_input:
                image = dist.shard_tensor(image, self._mesh, [dist.Shard(0)])

            with paddle.amp.auto_cast(level=level, dtype=self._dtype):
                out = layer(image)
                loss = loss_fn(out, label)

            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.step(opt)
            scaler.update()
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
        tol = 0.005
        self.set_random_seed(self._seed)
        (
            self.dp_loss_o1,
            self.dp_parameters_o1,
        ) = self.run_dynamic_amp(
            DemoNet('dp_demo_weight_O1'), 'O1', shard_input=True, run_dist=True
        )
        np.testing.assert_allclose(
            self.dp_loss_o1.numpy(),
            self.base_loss_o1.numpy(),
            rtol=tol,
            atol=tol,
        )
        for param, param_base in zip(
            self.dp_parameters_o1, self.base_parameters_o1
        ):
            np.testing.assert_allclose(
                param.numpy(), param_base.numpy(), rtol=tol, atol=tol
            )
            np.testing.assert_allclose(
                param.grad.numpy(), param_base.grad.numpy(), rtol=tol, atol=tol
            )

        self.set_random_seed(self._seed)
        (
            self.dp_loss_o2,
            self.dp_parameters_o2,
        ) = self.run_dynamic_amp(DemoNet('dp_demo_weight_O2'), 'O2')
        np.testing.assert_allclose(
            self.dp_loss_o2.numpy(),
            self.base_loss_o2.numpy(),
            rtol=tol,
            atol=tol,
        )
        for param, param_base in zip(
            self.dp_parameters_o2, self.base_parameters_o2
        ):
            np.testing.assert_allclose(
                param.numpy(), param_base.numpy(), rtol=tol, atol=tol
            )
            np.testing.assert_allclose(
                param.grad.numpy(), param_base.grad.numpy(), rtol=tol, atol=tol
            )

    def test_mp_demo_net(self):
        tol = 0.005
        self.set_random_seed(self._seed)
        mp_layer_o1 = dist.shard_layer(
            DemoNet("mp_demo_weight_O1"), self._mesh, self.shard_fn
        )
        (
            self.mp_loss_o1,
            self.mp_parameters_o1,
        ) = self.run_dynamic_amp(mp_layer_o1, 'O1', run_dist=True)
        np.testing.assert_allclose(
            self.mp_loss_o1.numpy(),
            self.base_loss_o1.numpy(),
            rtol=tol,
            atol=tol,
        )
        for param, param_base in zip(
            self.mp_parameters_o1, self.base_parameters_o1
        ):
            np.testing.assert_allclose(
                param.numpy(), param_base.numpy(), rtol=tol, atol=tol
            )
            np.testing.assert_allclose(
                param.grad.numpy(), param_base.grad.numpy(), rtol=tol, atol=tol
            )

        self.set_random_seed(self._seed)
        mp_layer_o2 = dist.shard_layer(
            DemoNet("mp_demo_weight_O2"), self._mesh, self.shard_fn
        )
        (
            self.mp_loss_o2,
            self.mp_parameters_o2,
        ) = self.run_dynamic_amp(mp_layer_o2, 'O2', run_dist=True)
        np.testing.assert_allclose(
            self.mp_loss_o2.numpy(), self.base_loss_o2.numpy(), rtol=tol
        )
        for param, param_base in zip(
            self.mp_parameters_o2, self.base_parameters_o2
        ):
            np.testing.assert_allclose(
                param.numpy(), param_base.numpy(), rtol=tol, atol=tol
            )
            np.testing.assert_allclose(
                param.grad.numpy(), param_base.grad.numpy(), rtol=tol, atol=tol
            )

    def run_test_case(self):
        if self._dtype == "bfloat16" and not paddle.amp.is_bfloat16_supported():
            return
        self.test_dp_demo_net()
        self.test_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetWithAmpForSemiAutoParallel().run_test_case()
