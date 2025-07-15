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
from semi_auto_parallel_dist_to_static_api import DemoNet, create_data_loader

import paddle
import paddle.distributed as dist
from paddle import nn


class TestSemiAutoParallelShardingStage1:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["dp"])
        self._multi_dim_mesh = dist.ProcessMesh(
            [[0, 1]], dim_names=["pp", "dp"]
        )

    def check_tensor_eq(self, a, b, rtol=1e-05, atol=0, verbose=True):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, verbose=verbose)

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

    def test_pure_sharding_stage_1(self):
        paddle.distributed.auto_parallel.set_mesh(self._mesh)
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        # shard the input by sharding degree
        batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
        # shard optimizer with stage 1 fn
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        opt = dist.shard_optimizer(opt, dist.ShardingStage1("dp", self._mesh))
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        self.check_tensor_eq(self.weight, linear.weight.numpy())
        self.check_tensor_eq(self.bias, linear.bias.numpy())

    def test_pure_sharding_multi_mesh_stage_1(self):
        paddle.distributed.auto_parallel.set_mesh(self._multi_dim_mesh)
        paddle.seed(self._seed)
        linear = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        # shard the input by sharding degree
        batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
        # shard optimizer with stage 1 fn
        opt = paddle.optimizer.AdamW(parameters=linear.parameters())
        opt = dist.shard_optimizer(
            opt, dist.ShardingStage1(sharding_mesh_dim="dp")
        )
        for _ in range(5):
            loss = linear(batch)
            loss.backward()
            opt.step()
            opt.clear_grad()
        self.check_tensor_eq(self.weight, linear.weight.numpy())
        self.check_tensor_eq(self.bias, linear.bias.numpy())

    def test_sharding_stage_1_to_static(self):
        paddle.distributed.auto_parallel.set_mesh(self._mesh)
        data_loader = create_data_loader()
        layer = DemoNet(self._mesh, "sharding_demonet")
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        opt = dist.shard_optimizer(opt, dist.ShardingStage1("dp", self._mesh))
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

    def test_sharding_stage_1_overlap_to_static(self):
        paddle.distributed.auto_parallel.set_mesh(self._mesh)
        data_loader = create_data_loader()
        layer = DemoNet(self._mesh, "sharding_demonet")
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        opt = dist.shard_optimizer(opt, dist.ShardingStage1("dp", self._mesh))
        loss_fn = nn.MSELoss()

        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self._mesh],
            shard_dims=0,
        )
        strategy = dist.Strategy()
        strategy.sharding.enable = True
        strategy.sharding.enable_overlap = True
        dist_model = dist.to_static(layer, dist_loader, loss_fn, opt, strategy)

        dist_model.train()
        for epoch in range(2):
            for batch_id, (image, label) in enumerate(dist_loader()):
                loss = dist_model(image, label)

    def test_pure_sharding_multi_mesh_stage_1_with_tensor_fusion(self):
        def run_sharding_test(enable_tensor_fusion):
            os.environ['FLAGS_enable_tensor_fusion'] = (
                '1' if enable_tensor_fusion else '0'
            )
            paddle.distributed.auto_parallel.set_mesh(self._multi_dim_mesh)
            paddle.seed(self._seed)
            model = paddle.nn.Linear(10, 10)
            batch = paddle.rand(shape=[10, 10])
            batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
            opt = paddle.optimizer.AdamW(parameters=model.parameters())
            opt = dist.shard_optimizer(
                opt, dist.ShardingStage1(sharding_mesh_dim="dp")
            )
            if enable_tensor_fusion:
                opt._enable_tensor_fusion()
            model, opt = paddle.amp.decorate(
                model, optimizers=opt, level='O2', master_grad=True
            )
            for _ in range(5):
                with paddle.amp.auto_cast(level='O2'):
                    loss = model(batch)
                    loss.backward()
                    opt.step()
                    opt.clear_grad()
            return loss.numpy()

        dist.init_parallel_env()
        loss_disable = run_sharding_test(enable_tensor_fusion=False)
        loss_enable = run_sharding_test(enable_tensor_fusion=True)
        self.check_tensor_eq(loss_disable, loss_enable)

    def test_pure_sharding_multi_mesh_stage_1_with_tensor_fusion_with_chip(
        self,
    ):
        dist.init_parallel_env()
        os.environ['FLAGS_enable_tensor_fusion'] = '1'
        paddle.distributed.auto_parallel.set_mesh(self._multi_dim_mesh)
        paddle.seed(self._seed)
        model = paddle.nn.Linear(10, 10)
        batch = paddle.rand(shape=[10, 10])
        batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
        clip = paddle.nn.ClipGradByGlobalNorm(1.0)
        opt = paddle.optimizer.AdamW(
            parameters=model.parameters(), grad_clip=clip
        )
        opt = dist.shard_optimizer(
            opt, dist.ShardingStage1(sharding_mesh_dim="dp")
        )
        opt._enable_tensor_fusion()
        model, opt = paddle.amp.decorate(
            model, optimizers=opt, level='O2', master_grad=True
        )
        for _ in range(5):
            with paddle.amp.auto_cast(level='O2'):
                loss = model(batch)
                loss.backward()
                opt.step()
                opt.clear_grad()

    def test_pure_sharding_multi_mesh_stage_1_with_sharding_overlap(self):
        def run_sharding_test(enable_sharding_overlap):
            os.environ['FLAGS_enable_tensor_fusion'] = '1'
            os.environ['FLAGS_enable_sharding_overlap'] = (
                '1' if enable_sharding_overlap else '0'
            )
            paddle.distributed.auto_parallel.set_mesh(self._multi_dim_mesh)
            paddle.seed(self._seed)
            model = paddle.nn.Linear(10, 10)
            batch = paddle.rand(shape=[10, 10])
            batch = dist.shard_tensor(batch, self._mesh, [dist.Shard(0)])
            opt = paddle.optimizer.AdamW(parameters=model.parameters())
            opt = dist.shard_optimizer(
                opt, dist.ShardingStage1(sharding_mesh_dim="dp")
            )
            opt._enable_tensor_fusion()
            if enable_sharding_overlap:
                opt._enable_sharding_overlap(model)
            model, opt = paddle.amp.decorate(
                model, optimizers=opt, level='O2', master_grad=True
            )
            for _ in range(5):
                with paddle.amp.auto_cast(level='O2'):
                    loss = model(batch)
                    loss.backward()
                    opt.step()
                    opt.clear_grad()
            return loss.numpy()

        dist.init_parallel_env()
        loss_disable = run_sharding_test(enable_sharding_overlap=False)
        loss_enable = run_sharding_test(enable_sharding_overlap=True)
        self.check_tensor_eq(loss_disable, loss_enable)

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.get_single_card_rst()
        self.test_pure_sharding_stage_1()
        self.test_sharding_stage_1_to_static()
        self.test_pure_sharding_multi_mesh_stage_1()
        self.test_sharding_stage_1_overlap_to_static()
        self.test_pure_sharding_multi_mesh_stage_1_with_tensor_fusion()
        self.test_pure_sharding_multi_mesh_stage_1_with_tensor_fusion_with_chip()
        self.test_pure_sharding_multi_mesh_stage_1_with_sharding_overlap()


if __name__ == '__main__':
    TestSemiAutoParallelShardingStage1().run_test_case()
