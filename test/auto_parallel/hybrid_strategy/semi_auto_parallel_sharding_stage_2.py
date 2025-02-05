# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from auto_parallel.semi_auto_parallel_dist_to_static_api import (
    DemoNet,
    create_data_loader,
)

import paddle
import paddle.distributed as dist
from paddle import nn


class TestSemiAutoParallelShardingStage2:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])

    def check_tensor_eq(self, a, b, rtol=1e-05, atol=0, verbose=True):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, verbose=verbose)

    def shard_layer_fn(self, layer_name, layer, process_mesh):
        layer.weight = dist.shard_tensor(
            layer.weight, process_mesh, [dist.Shard(1)]
        )
        layer.bias = dist.shard_tensor(
            layer.bias, process_mesh, [dist.Shard(0)]
        )

    def get_single_card_rst(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        self.weight = linear.weight.numpy()
        self.bias = linear.bias.numpy()

    def test_sharding_stage_2_with_mp(self):
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        linear = dist.shard_layer(linear, self._mesh, self.shard_layer_fn)
        batch = paddle.rand(shape=[10, 10])
        # shard the input by sharding degree
        batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
        # shard optimizer with stage 1 fn
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        opt = dist.shard_optimizer(opt, dist.ShardingStage2("x", self._mesh))
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        self.check_tensor_eq(self.weight, linear.weight.numpy())
        self.check_tensor_eq(self.bias, linear.bias.numpy())

    def test_sharding_stage_2_with_mp_to_static(self):
        data_loader = create_data_loader()
        layer = DemoNet(
            self._mesh, "sharding_with_mp_demonet", shard_weight=True
        )
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        opt = dist.shard_optimizer(opt, dist.ShardingStage2("x", self._mesh))
        loss_fn = nn.MSELoss()

        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self._mesh],
            shard_dims=0,
        )

        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt)

        dist_model.train()
        for epoch in range(2):
            for batch_id, (image, label) in enumerate(dist_loader()):
                loss = dist_model(image, label)

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.get_single_card_rst()
        self.test_sharding_stage_2_with_mp()
        self.test_sharding_stage_2_with_mp_to_static()


if __name__ == '__main__':
    TestSemiAutoParallelShardingStage2().run_test_case()
